import imgaug.augmenters as iaa

sometimes_0_5 = lambda aug: iaa.Sometimes(0.5, aug)
sometimes_0_2 = lambda aug: iaa.Sometimes(0.2, aug)


seq_grayscale = iaa.Sequential(
    [
        sometimes_0_2(iaa.GaussianBlur((0.0, 2.0))),
        sometimes_0_2(iaa.AverageBlur((3, 7))),
        sometimes_0_2(iaa.MedianBlur((3, 7))),
        sometimes_0_2(iaa.MotionBlur((3, 7))),
        sometimes_0_5(iaa.Add((-10, 10), per_channel=0.5)),
        sometimes_0_5(iaa.Multiply((0.75, 1.25), per_channel=0.5)),
        sometimes_0_5(iaa.GammaContrast((0.75, 1.25), per_channel=0.5)),
        # sometimes_0_5(iaa.SigmoidContrast((5, 10), per_channel=0.5)), # seemms to be too strong for grayscale images
        sometimes_0_5(iaa.LogContrast((0.75, 1.0), per_channel=0.5)),
        sometimes_0_5(iaa.LinearContrast((0.7, 1.3), per_channel=0.5)),
    ],
    random_order=True,
)

seq_color = iaa.Sequential(
    [
        sometimes_0_2(iaa.GaussianBlur((0.0, 2.0))),
        sometimes_0_2(iaa.AverageBlur((3, 7))),
        sometimes_0_2(iaa.MedianBlur((3, 7))),
        sometimes_0_2(iaa.MotionBlur((3, 7))),
        sometimes_0_2(iaa.BilateralBlur((1, 7))),
        sometimes_0_5(iaa.AddToHue((-15, 15))),
        sometimes_0_5(iaa.AddToSaturation((-15, 15))),
        # sometimes_0_5(iaa.Grayscale(alpha=(0.0, 0.2))), # seems to be very slow! and reduces speed of data pipeline by more than 10 % (remove?)
        sometimes_0_5(iaa.Add((-10, 10), per_channel=0.5)),
        sometimes_0_5(iaa.Multiply((0.75, 1.25), per_channel=0.5)),
        sometimes_0_5(iaa.GammaContrast((0.75, 1.25), per_channel=0.5)),
        sometimes_0_5(iaa.SigmoidContrast((5, 10), per_channel=0.5)),
        sometimes_0_5(iaa.LogContrast((0.75, 1.0), per_channel=0.5)),
        sometimes_0_5(iaa.LinearContrast((0.7, 1.3), per_channel=0.5)),
    ],
    random_order=True,
)

"""
Augmentation parameters similar to PyraPose by Stefan Thalhammer (https://github.com/sThalham/PyraPose).
"""
seq = iaa.Sequential(
    [
        # blur
        iaa.SomeOf(
            (0, 2),
            [
                iaa.GaussianBlur((0.0, 2.0)),
                iaa.AverageBlur(k=(3, 7)),
                iaa.MedianBlur(k=(3, 7)),
                iaa.BilateralBlur(d=(1, 7)),
                iaa.MotionBlur(k=(3, 7)),
            ],
        ),
        # color
        sometimes_0_5(iaa.AddToHueAndSaturation((-15, 15))),
        # iaa.SomeOf((0, 2), [
        #    #iaa.WithColorspace(),
        #     iaa.AddToHueAndSaturation((-15, 15)),
        #    #iaa.ChangeColorspace(to_colorspace[], alpha=0.5),
        #     iaa.Grayscale(alpha=(0.0, 0.2))
        # ]),
        # brightness
        iaa.OneOf(
            [
                iaa.Sequential(
                    [
                        iaa.Add((-10, 10), per_channel=0.5),
                        iaa.Multiply((0.75, 1.25), per_channel=0.5),
                    ]
                ),
                iaa.Add((-10, 10), per_channel=0.5),
                iaa.Multiply((0.75, 1.25), per_channel=0.5),
                iaa.FrequencyNoiseAlpha(
                    exponent=(-4, 0),
                    first=iaa.Multiply((0.75, 1.25), per_channel=0.5),
                    second=iaa.ContrastNormalization((0.7, 1.3), per_channel=0.5),
                ),
            ]
        ),
        # contrast
        iaa.SomeOf(
            (0, 2),
            [
                iaa.GammaContrast((0.75, 1.25), per_channel=0.5),
                iaa.SigmoidContrast(gain=(5, 10), cutoff=(0.25, 0.75), per_channel=0.5),
                iaa.LogContrast(gain=(0.75, 1), per_channel=0.5),
                iaa.LinearContrast(alpha=(0.7, 1.3), per_channel=0.5),
            ],
        ),
        # arithmetic
        iaa.SomeOf(
            (0, 3),
            [
                iaa.AdditiveGaussianNoise(scale=(0, 0.05), per_channel=0.5),
                iaa.AdditiveLaplaceNoise(scale=(0, 0.05), per_channel=0.5),
                iaa.AdditivePoissonNoise(lam=(0, 8), per_channel=0.5),
                iaa.Dropout(p=(0, 0.05), per_channel=0.5),
                iaa.ImpulseNoise(p=(0, 0.05)),
                iaa.SaltAndPepper(p=(0, 0.05)),
                iaa.Salt(p=(0, 0.05)),
                iaa.Pepper(p=(0, 0.05)),
            ],
        ),
        # iaa.Sometimes(p=0.5, iaa.JpegCompression((0, 30)), None),
    ],
    random_order=True,
)
