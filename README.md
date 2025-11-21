Sürdürülebilir Madencilik için Görüntü İşleme ile Otomasyon Sistemi

Bu proje görüntü işleme ile "yeraltında cevher ayıklama" araştırma önerisi olarak hazırlanmış ve protoriplenmiştir.
Yeraltı koşullarında kömür ve taş gibi malzemeleri OAK-1 Lite yapay zekâ kamerası ve YOLOv8n modeli kullanarak otomatik tanıyan, sınıflandıran ve servo kontrollü mekanik kol ile yönlendiren kompakt bir prototip geliştirilmiştir.

📸 Görüntü İşleme & Yapay Zeka

- OAK-1 Lite (Intel Myriad X VPU)

- YOLOv8n (gerçek zamanlı nesne tespiti)

- OpenCV / DepthAI

- Google Colab (A100 GPU)

- ONNX & blob model dönüştürme

- Roboflow ile veri etiketleme

🖥 Donanım

- Raspberry Pi 4 (8GB RAM)

- SG90 Servo Motorlar

- 3D Baskı Mekanik Kol

-  3S Li-Po Batarya

📦 Veri Seti Oluşturma ve OAK-1 Lite İçin .blob Modeli Üretme Süreci
Bu projede OAK-1 Lite kamera kullanıldığından, Google Colab’ta eğitilen YOLOv8 modelini doğrudan kullanmak mümkün değildir. Çünkü OAK-1 Lite, modelleri Intel Myriad X üzerinde çalıştırır ve bu donanım yalnızca OpenVINO tabanlı .blob formatını destekler.

Roboflow'dan alınan veri seti boyutu 320 olmalıdır, Resize → 320 × 320 (OAK-1 Lite performansı için optimum)
Google colabta eğitilen veri seti boyutunu 320 yapmayı unutmayın!

> from ultralytics import YOLO
> model = YOLO("yolov8n.pt")
> model.train(data="data.yaml", epochs=200, imgsz=320)

Burada elde edilen .pt formatını OAK-1 Lite üzeerinde kullanabilmek için ONNX formatı kabul eder sonrasında bunu .blob dosyasına dönüştürmek gerekir.
bknz  https://tools.luxonis.com/

