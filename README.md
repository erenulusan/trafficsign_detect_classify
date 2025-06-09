# trafficsign_detect_classify
YOLOv8 ile nesne tespiti ve EfficientNetV2 ile sınıflandırma yapan hibrit bir trafik işareti tanıma sistemi.

Proje Raporu: YOLOv8 ve EfficientNetV2 ile Hibrit Trafik İşareti Tanıma Sistemi
Bu projede, trafik işaretlerini tespit ve sınıflandırma görevlerini iki ayrı  model eğiterek bunları birbirine entegre ettim. Temel amaç, YOLOv8'in nesne tespiti (object detection) konusundaki hızı ve etkinliği ile EfficientNetV2 mimarisinin sınıflandırmadaki yüksek doğruluğunu bir araya getirerek robust ve gerçek dünya senaryolarına uygun bir yapı oluşturmaktı.

Sınıflandırma Modülü: Zorluklar ve Stratejik Değişiklikler
Projenin ilk fazı olan sınıflandırma modelini geliştirirken GTSRB veri setini kullandım.

İlk Denemeler ve "Domain Gap" Sorunu
İlk denemelerimde özel bir CNN mimarisi ve ardından ResNet50 ile transfer öğrenmeyi uyguladım. Her iki model de GTSRB veri setinin kendi içindeki validasyon setinde oldukça yüksek doğruluk oranlarına ulaşmasına rağmen, sisteme dışarıdan verdiğim gerçek dünya görsellerinde ciddi bir performans düşüşü yaşadılar. Bu durumun temel nedeninin klasik bir overfitting probleminden ziyade, eğitim ve test verileri arasındaki "domain farkı" (Domain Gap) olduğunu tespit ettim. GTSRB veri seti, büyük ölçüde ideal ve kontrollü koşullarda toplanmış bir "kaynak domaini" iken, gerçek dünya görselleri ise çok daha zorlu bir "hedef domaini" temsil ediyordu. 

Stratejik Çözüm
İlk olarak, eğittiğim modelde daha sert bi augmentation uyguladım fakat bu sefer model öğrenemedi.

Bu domain farkını kapatmak ve modelin genelleme kabiliyetini artırmak için yaklaşımımı temelden değiştirdim:

Gelişmiş Veri Artırma (Albumentations): ImageDataGenerator yerine, RandomBrightnessContrast, MotionBlur gibi daha gerçekçi ve zorlayıcı veri artırma teknikleri sunan Albumentations kütüphanesini kullandım.
Modern Mimarinin Gücü (EfficientNetV2): Model mimarisi olarak, hem parametre verimliliği hem de yüksek başarım sunan EfficientNetV2B0'ı seçtim.
Bu iki stratejik değişiklik sonucunda, model daha önce hatalı sınıflandırdığı tüm test görsellerini doğru bir şekilde tanımlayarak hedeflenen robustluğa ulaştı.

Nesne Tespiti Modülü: YOLOv8 ile Etkin Lokalizasyon
Sınıflandırma modeli hazır olduğunda, bu işaretleri bir görüntüden otomatik olarak bulacak tespit modülünü geliştirdim.

YOLOv8 ve Tek Sınıflı Yaklaşım
Bu görev için GTSDB veri setini kullanarak bir YOLOv8m modeli eğittim. Burada kritik bir tasarım kararı alarak, modeli 43 farklı işareti tanıyacak şekilde değil, tüm işaretleri tek bir "sign" sınıfı olarak tespit edecek şekilde yapılandırdım. Bu yaklaşım, tespit modelinin görevini (işaretin yerini bulma) basitleştirerek lokalizasyon hassasiyetini (precision ve recall) maksimize etti.

Entegre Sistem ve Çalışma Prensibi
Son aşamada, bu iki modül entegre bir pipeline içinde birleştirildi:

Tespit (Detection): Giriş görüntüsü üzerinde YOLOv8m ile inference çalıştırılarak trafik işaretlerinin koordinatları (bounding box) elde edilir.
Kırpma (Cropping): Tespit edilen her bir kutu, görüntüden bir İlgi Alanı (Region of Interest - ROI) olarak kırpılır.
Ön İşleme (Preprocessing): ROI, EfficientNetV2 modelinin giriş formatına (224x224 boyutlandırma, normalizasyon) uygun hale getirilir.
Sınıflandırma (Classification): Ön işlenmiş görüntü, fine-tune edilmiş EfficientNetV2 modeline verilerek nihai sınıf etiketi ve güven skoru elde edilir.
Sonuç ve Değerlendirme
Bu proje, "domain gap" gibi sorunları doğru teşhis etmenin ne kadar kritik olduğunu benim için bir kez daha kanıtladı. Sonuç olarak, YOLOv8'in hızlı tespit yeteneği ile doğru veri stratejileriyle eğitilmiş EfficientNetV2'nin yüksek doğruluklu sınıflandırma gücünü birleştiren, modüler ve robust bir sistem ortaya konmuştur.

Repo İçeriği ve Dosya Açıklamaları
Bu repoda bulunan script ve notebook'ların açıklamaları aşağıdadır:

siniflandirma.ipynb: GTSRB veri setiyle yapılan ilk sınıflandırma denemelerini içerir. Özel CNN ve ResNet50 modelleriyle yapılan ve "Domain Gap" sorunuyla karşılaşılan çalışmaları barındırır.
siniflandirma2.ipynb: Başarılı sınıflandırma modelinin geliştirildiği notebook'tur. EfficientNetV2 ve Albumentations kütüphanesi kullanılarak nihai sınıflandırıcı (.keras modeli) burada eğitilmiştir.
yolotrain.ipynb: YOLOv8m nesne tespit modelinin GTSDB veri seti üzerinde eğitildiği Google Colab notebook'udur.
convert.py: GTSDB veri setindeki .ppm formatlı görüntüleri, standart .jpg formatına dönüştürmek için kullanılan yardımcı script'tir.
split.py: GTSDB veri setini eğitim ve doğrulama setlerine ayıran ve YOLO formatına uygun etiket (.txt) dosyalarını oluşturan yardımcı script'tir.
detectandclassify.py: Projenin ana script'idir. Eğitilmiş YOLOv8 ve EfficientNetV2 modellerini kullanarak yeni bir görüntü üzerindeki trafik işaretlerini tespit eder ve sınıflandırır.
LICENSE: Projenin MIT lisansını içerir.
test1.png, test2.png, 00099.jpg, 00285.jpg: detectandclassify.py script'ini test etmek için kullanılabilecek örnek görsellerdir.
