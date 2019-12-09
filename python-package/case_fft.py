def fft_test(N = 28):
    dL = 0.02
    s = dL * dL / (N * N)

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    preprocess = transforms.Compose([
        #transforms.Resize(256),
        #transforms.CenterCrop(224),
        transforms.ToTensor(),
        #normalize
    ])
    img_raw = Image.open("E:/ONNet/data/MNIST/test_2.jpg")
    img_tensor = preprocess(img_raw)
    #img_tensor.unsqueeze_(0)
    print(img_tensor.shape, img_tensor.dtype)

    u0 = tf.cast(img_tensor, dtype=torch.complex64)
    print(u0.shape, H_f.shape);
    u1 = tf.fft2d(u0)
    u1 = H_f * u1
    u1 = tf.ifft2d(u1 * s) / s
    with tf.Session() as sess:  print(u1.eval())
    pass

#fft_test()