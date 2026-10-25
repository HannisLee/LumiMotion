"""以独立 NumPy/SciPy 公式复算指标，并完整解码四类视频。"""
import argparse
import json
from pathlib import Path
import imageio.v2 as imageio
import numpy as np
from scipy.ndimage import convolve1d


def independent_ssim(x, y):
    x, y = x.astype(np.float64), y.astype(np.float64)
    axis = np.arange(11, dtype=np.float64)-5
    gaussian = np.exp(-axis**2/(2*1.5**2))
    gaussian /= gaussian.sum()

    def filt(value):
        return convolve1d(convolve1d(value, gaussian, axis=-1, mode='constant', cval=0), gaussian, axis=-2, mode='constant', cval=0)

    mux, muy = filt(x), filt(y)
    vx, vy, covariance = filt(x*x)-mux*mux, filt(y*y)-muy*muy, filt(x*y)-mux*muy
    return float(np.mean(((2*mux*muy+.01**2)*(2*covariance+.03**2))/((mux*mux+muy*muy+.01**2)*(vx+vy+.03**2))))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', type=Path, required=True)
    args = parser.parse_args()
    evaluation = args.experiment/'evaluation'
    metrics = json.loads((evaluation/'metrics.json').read_text())
    records = []
    for row in metrics['frames']:
        with np.load(evaluation/'arrays'/(row['frame']+'.npz')) as data:
            assert all(np.isfinite(data[k]).all() for k in data.files)
            rgb, target, mask = (data[k].astype(np.float64) for k in ['rgb', 'target', 'mask'])
            error = (rgb-target)**2
            psnr = -10*np.log10(max(float(error.mean()), 1e-12))
            foreground = -10*np.log10(max(float((error*mask).sum()/(3*mask.sum())), 1e-12))
            ssim = independent_ssim(rgb, target)
        difference = dict(psnr=abs(psnr-row['psnr']), ssim=abs(ssim-row['ssim']), psnr_foreground=abs(foreground-row['psnr_foreground']))
        assert all(v < 1e-4 for v in difference.values()), (row['frame'], difference)
        records.append(dict(frame=row['frame'], psnr=psnr, ssim=ssim, psnr_foreground=foreground, absolute_difference=difference))
    videos = {}
    for name in ['rgb', 'alpha', 'normal', 'separation', 'albedo', 'roughness', 'error', 'gt']:
        reader = imageio.get_reader(str(evaluation/(name+'.mp4')))
        count = 0
        try:
            for frame in reader:
                assert frame.shape == (400, 400, 3)
                count += 1
        finally:
            reader.close()
        assert count == len(records)
        videos[name] = count
    result = dict(status='PASS', scope='指标公式、浮点数组与视频解码核验；不等于实际完整播放或画质验收',
                  numpy_scipy_independent=True, zero_padded_gaussian_window='11×11 sigma1.5', tolerance=1e-4,
                  frames=records, full_video_decoded_frames=videos,
                  metrics={k: float(np.mean([r[k] for r in records])) for k in ['psnr', 'ssim', 'psnr_foreground']})
    (args.experiment/'independent_metric_validation.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(dict(status='PASS', metrics=result['metrics'], videos=videos), ensure_ascii=False))


if __name__ == '__main__':
    main()
