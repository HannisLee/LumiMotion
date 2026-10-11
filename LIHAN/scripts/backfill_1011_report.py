#!/usr/bin/env python3
"""根据完成的补评测与实际图像目检，回填六组文档和总报告。"""
import csv,json,re
from pathlib import Path
import batch_1011 as b

KEYS=['albedo','roughness','gs_render','depth_normal','mesh','expected','median']
HEADER='| 编号／迭代 | Albedo MAE | Roughness MAE | GS Normal ° | Depth Normal ° | Mesh Normal ° | Expected Z | Median Z |'
RULE='未预先约定数值阈值，MAE不作任意门槛判定。最终FAILED依据必要训练阶段缺失，或GT对比中明确可见的投影残留／材质污染。视频完整解码与首、中、尾图像抽查分别记录，未声称逐帧人工检查全部视频。'

def value(x):return f'{x:.8f}' if isinstance(x,(float,int)) else str(x)

def record_hdr_completion(exp, metadata):
    import cv2
    from PIL import Image,ImageDraw
    out=Path(exp['output']);details=b.read_json(metadata)
    if details['status']!='PASS':return
    video=Path(details['output'])/'hdr/hdr.mp4';capture=cv2.VideoCapture(str(video));frames=[];count=0
    while True:
        ok,pixels=capture.read()
        if not ok:break
        if count in {0,75,149}:frames.append(Image.fromarray(cv2.cvtColor(pixels,cv2.COLOR_BGR2RGB)))
        count+=1
    capture.release()
    sheet=out/'inspection/hdr_retry_02__hdr.jpg';canvas=Image.new('RGB',(960,348),'#181818');draw=ImageDraw.Draw(canvas)
    for col,frame in enumerate(frames):
        frame.thumbnail((320,320));canvas.paste(frame,(col*320,28))
    draw.text((8,8),'HDR 2048 samples; video indices 0 / 75 / 149',fill='white');canvas.save(sheet)
    original=b.read_json(out/'inspection/artifacts.json')
    backup=out/'review_1011/original_artifacts_before_hdr.json'
    if not backup.exists():b.write_json(backup,original)
    decoded=all(r['解码帧数']==r['预期帧数'] for r in original['视频'])
    record=dict(原始产物记录=str(backup),原始产物核验=original['产物核验'],补渲染视频=str(video),
        补渲染解码帧数=count,代表图=str(sheet),完整视频总数=len(original['视频'])+1,
        完整渲染核验='PASS' if decoded and count==150 and len(original['视频'])+1>=6 else 'FAILED',
        说明='此结论只验收渲染文件完整性；Roughness和材质质量结论另记。')
    b.write_json(out/'inspection/combined_artifacts.json',record)

def main():
    config=b.read_json(b.CONFIG);experiments=config['experiments'];reviews=b.read_json(b.CONTROL/'visual_review.json')
    hdr_final=Path(experiments[4]['output'])/'review_1011/hdr_retry_02.json'
    hdr_done=hdr_final.exists()
    if hdr_done:record_hdr_completion(experiments[4],hdr_final)
    rows=[];states=[];sections=[];rgb=[]
    for exp in experiments:
        out=Path(exp['output']);index=exp['id'];review=reviews[str(index)];iteration=35000 if index<=4 else 55000
        if not (out/f'evaluation_{iteration}/mae_summary.json').exists():raise RuntimeError('评测仍未完成：'+str(index))
        if index<=5 and not(out/'review_1011/completed.json').exists():raise RuntimeError('补评测仍在进行：'+str(index))
        original=b.read_json(out/'review_1011/original_state.json')
        original_status='FAILED' if (out/'FAILED').exists() else original['status']
        state=b.read_json(out/'state.json');state.update(review_conclusion=review['结论'],reviewed_at=b.now(),review_reason=review['最终依据'])
        if index==6:state.update(status='FAILED',stage='训练评测完成；材质与重光照质量验收失败')
        elif index==5:state.update(status='FAILED',stage='55k训练完成；原始HDR失败；Roughness及材质质量验收失败')
        state['gpu']=None;state['child_pid']=None;b.write_json(out/'state.json',state)
        states.append(dict(experiment=index,dataset=exp['dataset'],method=exp['kind'],**state))
        metric_table=[HEADER,'|---|---|---|---|---|---|---|---|']
        for it in ([35000,55000] if index<=4 else [55000]):
            path=out/f'evaluation_{it}/mae_summary.json';metrics=b.read_json(path) if path.exists() else {}
            absent=not path.exists();record=dict(experiment=index,dataset=exp['dataset'],method=exp['kind'],iteration=it,
                status=metrics.get('status','UNAVAILABLE'),reason='原生Stage2首步失败，无55k检查点' if absent else '；'.join(metrics.get('errors',[])))
            vals=[]
            for key in KEYS:
                na=(it==35000 and key in {'albedo','roughness'}) or (index==6 and key=='roughness')
                item='N/A' if na else metrics.get(key,'未产出' if absent else '失败：GT超范围' if index==5 and key=='roughness' else '评测失败')
                record[key]=None if isinstance(item,str) else item;vals.append(value(item))
            rows.append(record);metric_table.append(f'| {index:02d}／{it} | '+' | '.join(vals)+' |')
        metricpath=out/f'evaluation_{iteration}/mae_summary.json';metrics=b.read_json(metricpath)
        if index<=4:
            log=(out/'logs/stage1.log').read_text(errors='replace')
            match=re.search(r'\[ITER 35000\] Evaluating reconstruct test: L1 ([\d.]+) PSNR ([\d.]+) SSIM ([\d.]+) LPIPS ([\d.]+)',log)
            if match:rgb.append(dict(experiment=index,iteration=35000,protocol='Stage1原始日志（已四舍五入；不与05/06直接比较）',psnr=float(match[2]),ssim=float(match[3]),lpips=float(match[4])))
        else:
            rgbfile=Path(exp['model'])/'renders_stage2/ours_55000/eval/metrics.json';data=b.read_json(rgbfile)
            rgb.append(dict(experiment=index,iteration=55000,protocol=data['protocol'],**data['metrics']))
        coverage=['| 指标 | 有效像素覆盖率 | 共同像素法线MAE ° |','|---|---|---|']
        for key in ['gs_render','depth_normal','mesh','expected','median']:
            item=metrics.get('details',{}).get(key,{})
            coverage.append('| '+key+' | '+value(100*item['coverage'])+'% | '+value(item.get('common_mae_deg') if item.get('common_mae_deg') is not None else 'N/A')+' |')
        source=exp['eval_source'];model=exp['model'];renderfolder=Path(model)/('renders_stage1_insights/ours_35000' if index<=4 else 'renders_stage2/ours_55000')
        actual=b.read_json(out/'review_1011/executed.json') if (out/'review_1011/executed.json').exists() else []
        original_commands=[json.loads(line) for line in (out/'commands/executed.jsonl').read_text().splitlines()]
        b.write_json(out/'review_1011/original_terminal_result.json',dict(status=original_status,
            evidence=str(out/'FAILED') if original_status=='FAILED' else str(out/'review_1011/original_state.json'),
            last_command=original_commands[-1]))
        commands=original_commands+actual
        required_audit=out/'review_1011/verify_all_mae_required.json'
        if required_audit.exists():commands.append(dict(step='PBR必需Roughness的独立复算',**b.read_json(required_audit)))
        retries=sorted((out/'review_1011').glob('hdr_retry*.json'))
        retry=retries[-1] if retries else out/'review_1011/hdr_retry.json'
        for item in retries:commands.append(dict(step='HDR补渲染 '+item.stem,**b.read_json(item)))
        if index==5 and not hdr_done:
            command_file=out/'review_1011/hdr_retry_02/command.json'
            if command_file.exists():
                pending=b.read_json(command_file)
                commands.append(dict(step='HDR第二次补渲染（运行中）',command=b.cmd(b.ROOT,'scripts.render_stage2',
                    '--model_path',exp['model'],'--load_iter',55000,'--task','hdr','--hdr_samples',2048,
                    '--hdr_filepath',pending['hdr_filepath'],'--output_path',pending['output_path']),
                    exit_code='RUNNING',log=str(out/'review_1011/hdr_retry_02.log'),
                    environment=dict(PYTHONPATH=str(b.CONTROL/'tools/nvdiffrast-sm86/extension'))))
        actualtext=['## 训练结果回填（'+b.now()+'）','',f'最终结论：**{review["结论"]}**。{review["最终依据"]}','','判定口径：'+RULE,'',
            f'- 原始训练／渲染执行结论：`{original_status}`；失败标记与退出码证据：`{out / "review_1011/original_terminal_result.json"}`；补评测开始时状态快照：`{out / "review_1011/original_state.json"}`。',
            f'- 实际有效检查点：{iteration}；model：`{model}`；Stage1冻结副本：`{exp["stage1_model"]}`。',
            f'- 原生输入source：`{exp["source"]}`；MAE输入source：`{source}`；渲染source由模型cfg_args读取。',
            f'- 渲染目录：`{renderfolder}`；评测目录：`{out / f"evaluation_{iteration}"}`。',
            '- 本次只评测与渲染已有检查点，未重启训练，未删除或覆盖原始失败结果。',
            f'- 目检拼图：`{out / "inspection/review_four_classes.png"}`；来源与视频帧索引：`{out / "inspection/review_four_classes_sources.json"}`。',
            f'- 完整视频解码、检查点有限性：`{out / "inspection/artifacts.json"}`；全15测试帧MAE独立复算：`{metricpath}`。','',
            '### 定量结果','']+metric_table+['','法线单位为度；深度是相机Z世界单位，单位比例与逐帧统计详见normal_depth/metrics.json。Albedo采用线性RGB前景核心区域，不作拟合后替代分数；Roughness为r参数。','','### 覆盖率与共同像素','']+coverage+['','### RGB、Alpha、Normal、分离与材质目检','']
        for key in ['RGB','Alpha','Normal','分离','材质与重光照']:
            actualtext.append('- **'+key+'**：'+review[key])
        actualtext+=['','目检范围：'+review['范围'],'','### 原始失败及本次补评测','']
        if index<=4:actualtext+=['原生Stage2在首个材质训练步调用nvdiffrast.texture失败，CUDA 209；没有55k检查点。原环境二进制只含sm_120内核，A40为sm_86。35000的几何分数仅代表Stage1，不能称为完成两阶段。']
        elif index==5:actualtext+=['训练正常保存55k，原始RGB／材质／转灯渲染完成后，HDR入口因同一CUDA架构不匹配失败。原始logs/render.log及其不完整渲染目录完整保留。第一次补渲染跨过纹理采样后暴露renderutils的模块载入错误，该失败目录也保留；修复load返回模块的缓存后再单独补渲染，最终状态见`'+str(retry)+'`。', '', 'Roughness测试已实际执行，但GT EXR去预乘后出现超出[0,1]的值，严格评测器拒绝，未夹取、未剔除异常后生成通过分数。15帧中48个越界像素（其中帧30有1个前景核心像素，r约1.0014025），审计见roughness_gt_audit.json。旧审计错误地将缺失Roughness视为可选，已增加PBR必需Roughness检查，整项MAE复算状态更正为FAILED；几何与Albedo独立数值保留。']
        else:actualtext+=['训练及全部适用MAE已完成；原来的AWAITING_VISUAL_REVIEW已完成实际图像目检。Lambertian没有Roughness参数，且HDR入口是无可见性漫反射球面积分；不能认作完整PBR。分离仅继承01 Stage1，不存在新训练的Stage2分离模块。']
        actualtext+=['','### 实际执行命令、退出码与日志','']
        for record in commands:
            if not record.get('command'):continue
            actualtext+=['#### '+record['step'],'',f'退出码：{record.get("exit_code", "未记录")}；日志：`{record.get("log", "见executed.json")}`。','']
            if record.get('environment'):actualtext+=['进程环境：`'+json.dumps(record['environment'],ensure_ascii=False)+'`。','']
            actualtext+=['```bash',b.shell_line(record['command']),'```','']
        rgbcurrent=rgb[-1];actualtext+=['### RGB测试指标','', '```json',json.dumps(rgbcurrent,ensure_ascii=False,indent=2),'```','', '### 保留产物与代表文件','']
        directional=Path(exp['model'])/'renders_stage2/ours_55000/directional/relight.json'
        if directional.exists():
            response=b.read_json(directional)
            actualtext+=['转灯每帧平均亮度范围：'+value(min(response['mean_brightness']))+'—'+value(max(response['mean_brightness']))+
                '；Alpha不变：'+str(response['alpha_unchanged'])+'。此项只证明有光照响应，不证明投影或材质正确。证据：`'+str(directional)+'`。','']
        artifacts=b.read_json(out/'inspection/artifacts.json')
        for record in artifacts.get('视频',[]):actualtext.append('- 视频：`'+record['文件']+'`；解码帧数：'+str(record['解码帧数'])+'；代表图：`'+record.get('代表图','无')+'`。')
        if retry.exists():actualtext.append('- 补渲染HDR：`'+str(Path(b.read_json(retry)['output'])/'hdr/hdr.mp4')+'`；统计：`'+str(retry)+'`。')
        combined=out/'inspection/combined_artifacts.json'
        if combined.exists():
            data=b.read_json(combined);actualtext.append('- 补齐后的完整渲染核验：'+data['完整渲染核验']+'；合计'+str(data['完整视频总数'])+'段视频；证据：`'+str(combined)+'`；HDR代表图：`'+data['代表图']+'`。')
        for path in sorted((out/f'evaluation_{iteration}').rglob('*comparison*.png')):
            if '0010' in path.name or '0080' in path.name or '0150' in path.name:actualtext.append('- GT对比：`'+str(path)+'`。')
        originaltext=(out/'review_1011/original_training_document.md').read_text().split('\n\n## 实际执行状态')[0]
        originaltext=originaltext.replace('当前为PENDING','启动前状态，实际验收见下方结果回填')
        content=originaltext+'\n\n'+'\n'.join(actualtext)+'\n'
        Path(exp['document']).write_text(content);(out/'README.md').write_text(content)
        b.write_json(out/'inspection/final_review.json',review)
        artifacts['人工目检']='已实际查看代表图，范围见final_review.json';artifacts['质量结论']=review['结论']
        b.write_json(out/'inspection/artifacts.json',artifacts)
        sections.append((exp,review))
    alltable=[HEADER,'|---|---|---|---|---|---|---|---|']
    for row in rows:
        vals=['N/A' if row['iteration']==35000 and k in {'albedo','roughness'} or row['experiment']==6 and k=='roughness' else value(row[k]) if row[k] is not None else '失败：GT超范围' if row['experiment']==5 and k=='roughness' else '未产出' for k in KEYS]
        alltable.append(f'| {row["experiment"]:02d}／{row["iteration"]} | '+' | '.join(vals)+' |')
    lines=['# 四个V2数据集六组训练与全部MAE总报告','',f'结果回填：{b.now()}；服务器ushas；Conda环境lumimotion-ushas。实验编号1011为用户指定，原始启动为2026-10-10，补评测为2026-10-11。','',
        '## 完成情况与最终结论','',
        '**整批结论：FAILED。** 01—04各自完成Stage1 35000，但原生Stage2首步因环境CUDA架构不匹配失败；05/06均完成累计55000。05原始HDR失败后单独补渲染，06原始训练评测正常完成。两组材质图均有明确可见的投影残留，不能报告完成可靠的材质／光照分解。','',
        '本次只补评现有检查点，未启动新训练。所有原始失败日志、状态、checkpoint、统计和渲染保留；原始文档另备份在各输出review_1011目录。','',RULE,'',
        '| 编号 | 数据集／方法 | 有效迭代 | 原始结果 | 最终验收 |','|---|---|---|---|---|']
    for exp,review in sections:
        index=exp['id'];origin=b.read_json(Path(exp['output'])/'review_1011/original_state.json')
        original_status='FAILED' if (Path(exp['output'])/'FAILED').exists() else origin['status']
        lines.append(f'| {index:02d} | {exp["dataset"]}／{exp["kind"]} | {35000 if index<=4 else 55000} | {original_status} | {review["结论"]} |')
    lines+=['','## 全部适用MAE','',
        '各个已评测检查点均使用完整15测试帧（10、20、…、150），135训练帧与测试帧分离。法线是角度（度）；深度为相机Z世界单位；Albedo为线性RGB；Roughness为原始r参数。Stage1的Albedo/Roughness及Lambertian Roughness为N/A；原生55k未产出，不能用Stage1代填。05 Roughness已尝试测试，但GT去预乘后超范围而失败，不能填N/A或虚构通过分数。','']+alltable
    lines+=['','MAE原始数组、15帧逐帧指标、预测Mesh训练视角清单和独立复算在各组evaluation_*/目录。汇总CSV／JSON：`'+str(b.CONTROL/'mae_summary.csv')+'`、`'+str(b.CONTROL/'mae_summary.json')+'`。各组文档列出覆盖率和共同有效像素法线分数。','',
        '05第二次HDR补渲染状态：'+(b.read_json(hdr_final)['status']+'；完整帧数'+str(b.read_json(hdr_final).get('decoded_frames',0)) if hdr_done else 'RUNNING；MAE与图像目检已回填，完整视频完成后再次更新路径与解码结果')+'。','',
        '## RGB结果与05／06比较','',
        '| 编号／迭代 | 全图PSNR | 全图SSIM | 全图LPIPS | 前景PSNR |','|---|---|---|---|---|']
    for row in rgb:lines.append(f'| {row["experiment"]:02d}／{row["iteration"]} | {value(row["psnr"])} | {value(row["ssim"])} | {value(row["lpips"])} | {value(row.get("psnr_foreground","原始日志未记"))} |')
    pbr=next(r for r in rows if r['experiment']==5);lambert=next(r for r in rows if r['experiment']==6)
    lines+=['','01—04的RGB分数来自原始训练日志，已被原程序四舍五入，仅作训练重建证据；05/06使用统一新渲染评测协议，有完整逐帧JSON及前景指标。不能直接按上述不同协议的分数比较原生35k与GT光源55k。','',
        f'05 Albedo MAE {pbr["albedo"]:.8f}，06 {lambert["albedo"]:.8f}；05降低{lambert["albedo"]-pbr["albedo"]:.8f}（{100*(1-pbr["albedo"]/lambert["albedo"]):.2f}%）。两组在同一静态DL、同一01冻结Stage1及135/15划分下可比较，但优化损失、光强约定和渲染方程不同，不能把差值单独归因于BRDF。','',
        'GT对比图显示，两组预测Albedo均残留GT Albedo不存在的黑色投影；05另有局部颜色噪声，06地面色块偏白。DL地面预测法线出现与投影／色块区域相近的起伏；颜色或阴影被几何吸收是可能解释，具体因果仍需额外实验确认。以上可见缺陷是质量验收失败的依据。','',
        '01—04的7段Stage1视频均完整150帧，检查点为有限值；产物审计仍FAILED是由于缺少Stage2材质／HDR等输出，不能解释为已生成视频损坏。06的6段视频通过完整解码。05原始5段视频及单独HDR的解码状态见各组产物JSON。','',
        '## CUDA失败根因与保留现场','',
        '原环境`_nvdiffrast_c`的cuobjdump列出五个内核仅为sm_120；服务器A40设备架构为sm_86。Conda CUDA driver_types.h将209定义为cudaErrorNoKernelImageForDevice，因此01—04的首步环境纹理采样、05的HDR纹理采样报同一错误。不是OOM，也不是完整原生Stage2收敛失败。','',
        '新旧扩展SHA256、完整cuobjdump命令和内核列表、四卡实际compute_cap查询：`'+str(b.CONTROL/'cuda_architecture_audit.json')+'`。','',
        '已从本机与安装包相同的官方来源提交253ac4fcea7de5f396371124af597e6cc957bfae编译隔离sm_86扩展，仅补HDR进程设置PYTHONPATH使用；未替换共享Conda里的原模块。源码、编译命令与日志：`'+str(b.CONTROL/'tools/nvdiffrast-sm86')+'`、`'+str(b.CONTROL/'logs/nvdiffrast_sm86_build.log')+'`。','',
        '第一次05补HDR跨过纹理采样后，又暴露renderutils已成功编译却重新import失败的问题。已改为直接缓存torch.utils.cpp_extension.load返回的模块，未改变着色数值；第二次尝试使用新目录，第一次补渲染失败结果仍保留。','',
        '01—04没有55000 checkpoint，完整原生Stage2及对应材质MAE仍缺失。此次回填不能解除这一实验缺口。','',
        '静态DL的GT导出命令虽然退出0，但Roughness浮点内容验证失败：15帧48个去预乘越界像素，帧30有1个前景核心越界像素。原始ready.json只代表导出命令执行完毕，不能当作GT数值验证通过。原始EXR保留，新内容审计位于`'+str(Path(experiments[4]['gt'])/'roughness_validation.json')+'`；05不能报告Roughness MAE。','',
        '## 各组四类目检与证据','']
    for exp,review in sections:
        lines+=['### '+f'{exp["id"]:02d} '+exp['dataset']+'／'+exp['kind'],'',
                '- RGB：'+review['RGB'],'- Alpha：'+review['Alpha'],'- Normal：'+review['Normal'],'- 分离：'+review['分离'],
                '- 材质与重光照：'+review['材质与重光照'],'- 最终'+review['结论']+'：'+review['最终依据'],
                '- 实际查看范围：'+review['范围'],
                '- [训练文档]('+exp['document']+')；[实验README]('+str(Path(exp['output'])/'README.md')+')；[代表图]('+str(Path(exp['output'])/'inspection/review_four_classes.png')+')。','']
    lines+=['## 评测约定与限制','',
        '- 输入Alpha沿用albedo-alpha；GT材质RGB、法线、深度、roughness、网格未用于训练监督。GT方向仅05/06使用。四组输入600帧审计PASS，源哈希和reader相机误差见input_audit.json。',
        '- 预测Mesh逐测试时间只融合135训练相机，GT网格不参与预测重建；法线覆盖率及共同像素统计保留。',
        '- 05使用两个单位白SUN方向与固定π光强倍率；06对重合SUN方向合并，并只用训练RGB标定固定非负强度；均未读取GT能量作强度监督。',
        '- EL原场景为双POINT，不能称为HDRI训练。原生static仍有时变形变网络，故每个测试时刻独立重建与评测。',
        '- 05／06分离模块来自01 Stage1，不是新的Stage2分离训练；继承分离缺陷不可归因于本次材质训练。',
        '- 06重光照积分为无遮挡漫反射，不包含镜面或间接光；方向光角度0是输入已知GT方向的结果，不是学习光源的精度。',
        '- 所有失败结果保留。每卡最多2任务，已有其他用户GPU进程计入调度。','',
        '共享资料同步：按AGENTS.md把LIHAN与LIHAN.md同步全10分支；同步证据保存在`'+str(b.CONTROL/'shared_sync.json')+'`。','']
    b.REPORT.write_text('\n'.join(lines));(b.CONTROL/'README.md').write_text('\n'.join(lines))
    b.write_json(b.CONTROL/'mae_summary.json',rows);b.write_json(b.CONTROL/'batch_status.json',states);b.write_json(b.CONTROL/'rgb_summary.json',rgb)
    with (b.CONTROL/'mae_summary.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    b.write_json(b.CONTROL/'final_backfill.json',dict(time=b.now(),status='COMPLETED' if hdr_done else 'REPORT_WRITTEN_HDR_RUNNING',batch_conclusion='FAILED',training_restarted=False,report=str(b.REPORT)))

if __name__=='__main__':main()
