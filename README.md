# NafNet deblur CoreML model

Transformed to CoreML [NAFNet](https://github.com/megvii-research/NAFNet) model for deblurring.
Normalization, preprocessing and postprocessing were integrated to network graph. FP8 quantized model is provided
[weights/nafnet_fp8.mlmodel](weights/nafnet_fp8.mlmodel).
Original model is [NAFNet-REDS-width64](https://github.com/megvii-research/NAFNet#results-and-pre-trained-models).

Model accepts given sizes:
```python
[
    (256, 256), (512, 512), (768, 768), (1024, 1024),
    (1280, 1280), (1536, 1536), (1792, 1792), (2048, 2048), (2304, 2304),
    (2560, 2560), (2816, 2816), (3072, 3072), (3328, 3328), (3584, 3584),
    (3840, 3840), (4096, 4096), (4352, 4352), (4608, 4608), (4864, 4864)
]
```

Execution script pads provided images to the nearest size listed above.

![result](results/res.png)

## Installation

1. Create Python virtual environment:
```shell
virtualenv -p python3.9 .venv && source .venv/bin/activate
```

2. Install all requirements:
```shell
pip install -r requirements.txt
```
3. Use the project :tada:

## Usage

To run use command below:
```shell
python run_nafnet_deblur.py --data-root images
```

Model inference will work only on MacOS. Also, model loading will take a LOT of time.
Generally, it is not suitable for mobile devices...

## Model

Model input is RGB Image. Model output is float32
image of shape (H, W, 3). Each pixel value of output varies from 0.0f to 255.0f.

Input node name is `image`, output - `result`.

## Built With

* [coremltools](https://github.com/apple/coremltools) - The NNs inference framework
* [OpenCV](https://opencv.org/) - Images processing framework


## License

This project is licensed under the MIT License – see the [LICENSE](LICENSE) file for details

## Authors

* **Vadim Titko** aka *Vadbeg* -
[LinkedIn](https://www.linkedin.com/in/vadtitko/) |
[GitHub](https://github.com/Vadbeg)
