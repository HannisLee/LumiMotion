"""从冻结及独立评测产物撰写中文报告；不更改模型、参数或预测。"""
import argparse
import json
from pathlib import Path
import shlex


def read(path):return json.loads(Path(path).read_text())


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--control',type=Path,required=True);a=p.parse_args()
    out=a.control.resolve();root=Path(__file__).resolve().parents[2]
    result=read(out/'evaluation/results.json');selection=read(out/'baseline_selection.json');config=read(out/'fitting/method_config.json')
    extraction=read(out/'extraction/extraction.json');baseline_review=read(out/'baseline_review.json');method_review=read(out/'method_review.json')
    model=Path(selection['model']);iteration=selection['selected_iteration'];final=result['methods'][result['final_method']]
    lines=['# 静态ELDL纯RGB未知方向光初始化实验','',f'最终结论：**{result["status"]}**。冻结最终方法E4的Mean Angular Error为{final["mean_deg"]:.4f}°，目标为≤5°。',
       '', '本轮只做方向光初始化，无Stage2训练。GT仅在模型、方法、全部预测冻结并通过两次无GT目检后独立读取一次；评测后不修改预测或参数。',
       '', '## 数据与基线','',
       '原始source：`data/LH-data/static/LH-static-jumpingjacks-ELDL-V3`。原RGBA文件中仅RGB三通道参与任何分割、监督和光度优化；Alpha不参与监督，未打开albedo、normal、depth、material、shadow等GT文件。RGB边界颜色连通推断mask，误差未经真值分割校正。',
       '',f'训练source：`{out/"source_train"}`（117 TRAIN，DEV18/TEST15隔离）；观测source：`{out/"source_all"}`。',
       '',f'固定model：`{model}`；iteration：**{iteration}**。几何从随机100k点纯RGB训练，形变头零值冻结，pred_color=False；既有S0/B00因使用albedo来源Alpha排除。',
       '', '按DEV18固定RGB推断前景mask>0.5（不腐蚀，不依模型alpha裁分母）的前景PSNR主选；0.1dB内依次比较内部法线cos、覆盖、全图LPIPS。SSIM及LPIPS均为全图黑背景作用域，背景可能提高这些分数，不作为前景质量替代。',
       '', '|保存点|前景PSNR|全图PSNR|全图SSIM|全图LPIPS|模型Alpha覆盖|GS/模型深度法线cos|邻域法线cos|Gaussian数|',
       '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for item in selection['candidates']:
        m=item['metrics'];lines.append(f'|{item["iteration"]}|{m["foreground_psnr"]:.4f}|{m["psnr"]:.4f}|{m["ssim"]:.5f}|{m["lpips"]:.5f}|{m["predicted_alpha_coverage"]:.4f}|{m["normal_depth_cosine"]:.4f}|{m["normal_neighbor_cosine"]:.4f}|{item["point_count"]}|')
    geom=selection['selected']['metrics']
    if 'near_horizontal_model_fraction' in geom:
        lines += ['',f'冻结模型近水平面（仅模型abs(n_z)>0.95，非GT平台标签）比例{geom["near_horizontal_model_fraction"]:.3f}，其GS/模型深度法线cos {geom["near_horizontal_normal_depth_cosine"]:.4f}，其余有效面cos {geom["other_model_normal_depth_cosine"]:.4f}。']
    lines += ['', '## 光向与光度结果','', '|方法|Mean°|Median°|P90°|≤5°帧数|≤5°比例|','|---|---:|---:|---:|---:|---:|']
    for method,m in result['methods'].items():
        count=m.get('within_5deg_count',m.get('le_5_count',m.get('count_le_5_deg','见results.json')))
        fraction=m.get('within_5deg_ratio',m.get('within_5deg_fraction',m.get('le_5_fraction',m.get('fraction_le_5_deg',None))))
        ratio=str(fraction) if fraction is None else f'{fraction:.2%}'
        lines.append(f'|{method}|{m["mean_deg"]:.4f}|{m["median_deg"]:.4f}|{m["p90_deg"]:.4f}|{count}|{ratio}|')
    lines += ['', 'E0固定world [0,0,1]；E1绝对亮度法线票；E2 Q20/Q95相对亮度法线票；E3跨帧共享rho/b鲁棒光度拟合；E4按固定空间留出选择法线均衡/阴影降权四种组合，若全关最优则回退E3。所有灯向单位化，方向光强固定1，无逐帧强度。',
      '', f'E4最终有限选择：{config["choices"]["E4"]}。该选择发生于GT读取前；E4与E3完全相同时即表示未获得改进。',
      '', 'E1—E4共用可靠对应与可观测性筛选（至少8次观测、Q95−Q20>0.015），并保持相同空间留出分割；E1是在该共同表面集上的绝对亮度投票，其投票信号不使用相对亮度。E2改用Q20/Q95归一化投票。该共同筛选用于控制变量，不把E1误称为全部原始单帧像素上的投票。',
      '', '百分位图高亮与实际fit表面/阈值一致；光度残差图为固定预测方向后逐表面重新OLS标定rho/b的诊断残差，不能当作内部Huber优化损失或空间留出CV残差。',
      '', '共享b和逆sRGB均为方法近似：导出源码不能严格证明所有EL贡献恒定，也未在RGB渲染入口固定view transform/exposure/gamma。没有使用GT方向挑选gamma或颜色变换。',
      '', f'可靠Gaussian观测：{extraction["observations"]}；拟合表面：{config["fit_gaussians"]}；固定空间留出表面：{config["heldout_gaussians"]}。缺观测回退帧：{config["missing_frames"]}。',
      '', f'法线二阶矩特征值：{config["normal_covariance_eigenvalues"]}；条件数：{config["normal_covariance_condition_number"]:.3f}。',
      '', f'无GT空间留出clipped MAE（上限0.2）：{config["heldout_photometric_clipped_mae"]}。该值仅衡量RGB光度解释，不等价灯向精度。',
      '', f'Bootstrap预测稳定性（对原预测角差，非GT角误差）：{config["bootstrap_stability"]}。',
      '', '## 完整命令及日志','']
    commands=[('',command) for command in read(out/'preparation_commands.json')]
    train_command=model.parent/'command.json'
    if train_command.exists():commands.append(('CUDA_VISIBLE_DEVICES=1 ',read(train_command)))
    for path in sorted((out/'logs').glob('*_command.json')):commands.append(('CUDA_VISIBLE_DEVICES=2 ',read(path)))
    for prefix,command in commands:lines += ['```bash',prefix+shlex.join(command),'```','']
    lines += [f'训练日志：`{model.parent/"train.log"}`。渲染/提取/拟合/冻结/一次评测日志：`{out/"logs"}`。监督状态：training_process.json、pipeline_state.json。',
      '',f'渲染输出：`{out/f"baseline_dev_{iteration}"}`与`{out/"extraction"}`；拟合诊断：`{out/"fitting"}`；独立角度评测：`{out/"evaluation"}`。',
      '', '## 四类可视化目检','', '以下结论在GT读取前已落盘，完整审查见baseline_review.json与method_review.json。','',
      '```json',json.dumps(dict(baseline=baseline_review,method=method_review),ensure_ascii=False,indent=2),'```','',
      '代表图片：rgb_mask_preview.jpg；baseline_dev_*/rgb_*.png、normal_*.png、coverage_*.png；extraction/correspondence_*.png；fitting/percentile_residual_*.png、photometric_residual_distribution.png、directions_no_gt.png；evaluation/angular_error_curve.png、angular_error_cdf.png、direction_components.png、failure_cases.png。',
      '', '本轮没有导出视频，视频路径不适用；全部代表图路径如上。逐帧预测与GT见evaluation/per_frame.csv（实际文件名以目录为准），误差失败案例见evaluation/failure_cases.json。',
      '', '## 限制与失败解释','',
      'RGB重建允许SH解释视角与光照变化，准确外观不保证准确物理法线。模型法线及对应受表面厚度、重叠Gaussian、遮挡和RGB推断轮廓影响。Lambertian+共享表面环境不能完全描述复杂环境、高光、阴影及相机视角变化。法线分布不均和材质/环境自由度还会导致不同灯向具有接近光度残差。若估计法线与灯向共同发生世界系旋转，点积光度可以近乎不变而绝对光向偏离；因此低留出残差或bootstrap稳定性均不能证明绝对角度达标。人工压力验证仅说明这些局限，不参与真实数据GT调参。失败帧排序用于解释局限，未据此修改预测。',
      '',f'冻结预测SHA256：`{result["predictions_sha256"]}`。方法、配置、模型、RGB观测与环境来源身份见freeze_manifest.json及code_environment_receipt.json。所有失败冒烟、检查点、渲染、统计和日志保留。',
      '',f'最终结论：**{result["status"]}**。']
    (out/'README.md').write_text('\n'.join(lines)+'\n')
    report=root/'LIHAN/DOC/报告/1030-04-静态ELDL纯RGB未知方向光初始化实验报告.md'
    report.write_text('\n'.join(lines)+'\n')
    goal=root/'LIHAN/DOC/GOAL/1031-01-光照幅度Stage2方向光初始化.md'
    text=goal.read_text();start=text.index('## 当前进度');end=text.index('## 实验结果与结论')
    text=text[:start]+'## 当前进度\n\n训练、基线筛选、E0—E4、无GT目检、冻结及一次独立GT评测均已完成；结果与失败证据完整保留。没有运行Stage2，也没有使用GT反向调参。\n\n'+text[end:]
    start=text.index('## 实验结果与结论')
    next_section=text.find('\n## ',start+len('## 实验结果与结论'))
    history=text[next_section:] if next_section>=0 else ''
    text=text[:start]+f'## 实验结果与结论\n\n最终方法E4，Mean {final["mean_deg"]:.4f}°，Median {final["median_deg"]:.4f}°，P90 {final["p90_deg"]:.4f}°。验收结论：**{result["status"]}**。\n\n完整实验与全部命令：`{out/"README.md"}`；中文报告：`{report}`。\n'+history
    goal.write_text(text)
    print('中文报告生成完成：'+result['status'])


if __name__=='__main__':main()
