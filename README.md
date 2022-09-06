#	Các khái niệm cơ bản
**Trí tuệ nhân tạo** (Artificial intelligence  – viết tắt là AI) là một ngành thuộc lĩnh vực khoa học máy tính (Computer science). Là trí tuệ do con người lập trình tạo nên với mục tiêu giúp máy tính có thể tự động hóa các hành vi thông minh như con người.

Trí tuệ nhân tạo khác với việc lập trình logic trong các ngôn ngữ lập trình là ở việc ứng dụng các hệ thống học máy (machine learning) để mô phỏng trí tuệ của con người trong các xử lý mà con người làm tốt hơn máy tính.

**Machine Learning** là một tập con của AI. Theo định nghĩa của Wikipedia Machine Learning là một lĩnh vực nhỏ của Khoa Học Máy Tính, nó có khả năng tự học hỏi dựa trên dữ liệu đưa vào mà không cần phải được lập trình cụ thể.

**Neural Networks** (NNs): Mạng nơ ron nhân tạo là một nhánh của ML, với ý tưởng từ mạng nơ ron của hệ thần kinh xây dựng lên một tập hợp các nút liên kết với nhau được xem như tế bào thần kinh, qua NN này máy tính sẽ có khả năng học.

**Deep Learning** (DL): Học sâu là một phần mở rộng của NN, kết hợp nhiều NN tạo thành các kiến trúc multi-layer NN, ở đó một NN là một layer, làm cho việc học máy sẽ hiệu quả hơn nhiều lần. Các kiến trúc DL phổ biến như: Deep Neural Networks (DNNs), Convolutional Neural Networks (CNNs), Recurrent Neural Networks (RNNs), Generative Adversarial Networks (GAN),…

**Computer Vision** là một lĩnh vực trong Artificial Intelligence và Computer Science (Trí tuệ nhân tạo và Khoa học máy tính) nhằm giúp máy tính có được khả năng nhìn và hiểu giống như con người.

# Các framework phổ biến hiện nay
##	Tensorflow

TensorfFlow là thư việc mã nguồn mở được xây dựng và phát triển bởi Google Brain. Thư việc sử dụng data flow graphs để tính toán, hỗ trợ API cho python, C++,…

Ưu điểm:
-	Thư viện đa dạng cho lập trình dataflow, nghiên cứu và phát triển DL
-	Có thế tính toán trên CPU/GPU, mobile,..
-	Là công cụ phổ biến nhất trong cộng đồng nghiên cứu DL.
-	Xử lý hiệu quả cho tính toán mảng đa chiều.

Nhược điểm:
-	API khó sử dụng để tạo DL model.

##	Keras 

Keras là thư việc Python giống như TensorFlow là công cụ hỗ trợ DL. Có thể thực thi trên CPU và GPU. Được phát triển với các nguyên tắc thân thiện với người dùng.

Ưu điểm:
-	Mã nguồn mở, nhanh.
-	Phổ biến, tài liệu tham khảo đầy đủ
-	Nhẹ nhàng và dễ sử dụng

Nhược điểm:
-	GPU không làm hết hiệu suất 100%
-	Kém linh hoạt, không phù hợp cho nghiên cứu kiến trúc mới.

##	Pytorch 
PyTorch là thư viện DL Python dành cho GPU. Được phát triển bởi Facebook. PyTorch được viết bằng python, C và CUDA.

Ưu điểm:
-	Hỗ trợ Dynamic computational graph
-	Hỗ trợ tự động phân biệt cho NumPy và SciPy
-	Sử dụng ngôn ngữ Python cho phát triển
-	Hỗ trợ ONNX

Nhược điểm:
-	Không hỗ trợ trên mobile

#	Convolution Neural Network

![CNN](https://thanhvie.com/wp-content/uploads/2020/08/introduction.jpeg)

**Mạng nơron tích chập** (còn gọi là ConvNet / CNN) là một thuật toán Deep Learning có thể lấy hình ảnh đầu vào, gán độ quan trọng (các trọng số - weights và độ lệch - bias có thể học được) cho các đặc trưng/đối tượng khác nhau trong hình ảnh và có thể phân biệt được từng đặc trưng/đối tượng này với nhau.

## Một số thành phần CNN
**Lớp tích chập ( Convolution layer)**

Tích chập là lớp đầu tiên để trích xuất các tính năng từ hình ảnh đầu vào. Tích chập duy trì mối quan hệ giữa các pixel bằng cách tìm hiểu các tính năng hình ảnh bằng cách sử dụng các ô vương nhỏ của dữ liệu đầu vào. Nó là 1 phép toán có 2 đầu vào như ma trận hình ảnh và 1 bộ lọc hoặc hạt nhân.

Sự kết hợp của 1 hình ảnh với các bộ lọc khác nhau có thể thực hiện các hoạt động như phát hiện cạnh, làm mờ và làm sắc nét bằng cách áp dụng các bộ lọc. Ví dụ dưới đây cho thấy hình ảnh tích chập khác nhau sau khi áp dụng các Kernel khác nhau.

![Filter](https://images.viblo.asia/470abaeb-08f4-4b0c-98a9-873bf5764f2a.png)

**Pooling** 

Lớp pooling sẽ giảm bớt số lượng tham số khi hình ảnh quá lớn. Không gian pooling còn được gọi là lấy mẫu con hoặc lấy mẫu xuống làm giảm kích thước của mỗi map nhưng vẫn giữ lại thông tin quan trọng. Các pooling có thể có nhiều loại khác nhau:

# Một số mô hình DL cho Computer Vision
## LeNet-5

![LeNet](https://phamdinhkhanh.github.io/assets/images/20200531_CNNHistory/pic4.png)

LeNet-5 là kiến trúc đầu tiên áp dụng mạng tích chập 2 chiều của giáo sư Yan Lecun, cha đẻ của kiến trúc CNN. Model ban đầu khá đơn giản và chỉ bao gồm 2 convolutional layers + 3 fully-connected layers. Mặc dù đơn giản nhưng nó có kết quả tốt hơn so với các thuật toán machine learning truyền thống khác trong phân loại chữ số viết tay như SVM, kNN.

Trong kiến trúc mạng nơ ron đầu tiên, để giảm chiều dữ liệu, Yan Lecun sử dụng Sub-Sampling Layer là một Average-Pooling Layer (các layer nhằm mục đích giảm chiều dữ liệu mà không thay đổi đặc trưng chúng ta còn gọi là Sub-Sampling Layer). Kiến trúc này khó hội tụ nên ngày nay chúng được thay thế bằng Max-Pooling.

Đầu vào của mạng LeNet có kích thước nhỏ (chỉ 32x32) và ít layers nên số lượng tham số của nó chỉ khoảng 60 nghìn.

## AlexNet

![AlexNet](https://phamdinhkhanh.github.io/assets/images/20200531_CNNHistory/pic5.png)

AlexNet là mạng CNN được giới thiệu vào năm 2012 bởi Alex Krizhevsky và dành chiến thắng trong cuộc thi ImageNet với cách biệt khá lớn so với vị trí thứ hai. Lần đầu tiên Alex net đã phá vỡ định kiến trước đó cho rằng các đặc trưng được học từ mô hình sẽ không tốt bằng các đặc trưng được tạo thủ công (thông qua các thuật toàn SUFT, HOG, SHIFT). Ý tưởng của AlexNet dựa trên LeNet của Yan Lecun và cải tiến ở các điểm:
-	Tăng kích thước đầu vào và độ sâu của mạng.
-	Sử dụng các bộ lọc (kernel hoặc filter) với kích thước giảm dần qua các layers để phù hợp với kích thước của đặc trưng chung và đặc trưng riêng.
-	Sử dụng local normalization để chuẩn hóa các layer giúp cho quá trình hội tụ nhanh hơn.

Ngoài ra mạng còn cải tiến trong quá trình optimizer như:
-	Lần đầu tiên sử dụng activation là ReLU (Rectified Linear Unit) thay cho Sigmoid. ReLU là hàm có tốc độ tính toán nhanh nhờ đạo hàm chỉ có 2 giá trị {0, 1} và không có lũy thừa cơ số e như hàm sigmoid nhưng vẫn tạo ra được tính phi tuyến (non-linear).
-	Sử dụng dropout layer giúp giảm số lượng liên kết neural và kiểm soát overfitting.
-	Qua các layers, kích thước output giảm dần nhưng độ sâu tăng dần qua từng kernel.

##	VGG-16

![VGG-16](https://phamdinhkhanh.github.io/assets/images/20200531_CNNHistory/pic8.png)

Với VGG-16, quan điểm về một mạng nơ ron sâu hơn sẽ giúp ích cho cải thiện độ chính xác của mô hình tốt hơn. Về kiến trúc thì VGG-16 vẫn dữ các đặc điểm của AlexNet nhưng có những cải tiến:
-	Kiến trúc VGG-16 sâu hơn, bao gồm 13 layers tích chập 2 chiều (thay vì 5 so với AlexNet) và 3 layers fully connected.
-	Lần đầu tiên trong VGG-16 chúng ta xuất hiện khái niệm về khối tích chập (block). Đây là những kiến trúc gồm một tập hợp các layers CNN được lặp lại giống nhau. Kiến trúc khối đã khởi nguồn cho một dạng kiến trúc hình mẫu rất thường gặp ở các mạng CNN kể từ đó.
-	VGG-16 cũng kế thừa lại hàm activation ReLU ở AlexNet.
-	VGG-16 cũng là kiến trúc đầu tiên thay đổi thứ tự của các block khi xếp nhiều layers CNN + max pooling thay vì xen kẽ chỉ một layer CNN + max pooling.
-	VGG-16 chỉ sử dụng các bộ lọc kích thước nhỏ 3x3 thay vì nhiều kích thước bộ lọc như AlexNet. Kích thước bộ lọc nhỏ sẽ giúp giảm số lượng tham số cho mô hình và mang lại hiệu quả tính toán hơn. 

Mạng VGG-16 sâu hơn so với AlexNet và số lượng tham số của nó lên tới 138 triệu tham số. Đây là một trong những mạng mà có số lượng tham số lớn nhất. Kết quả của nó hiện đang xếp thứ 2 trên bộ dữ liệu ImageNet validation ở thời điểm public. Ngoài ra còn một phiên bản nữa của VGG-16 là VGG-19 tăng cường thêm 3 layers về độ sâu.
##	ResNet
![ResNet](https://phamdinhkhanh.github.io/assets/images/20200531_CNNHistory/pic11.png)

ResNet là kiến trúc được sử dụng phổ biến nhất ở thời điểm hiện tại. ResNet cũng là kiến trúc sớm nhất áp dụng batch normalization. Mặc dù là một mạng rất sâu khi có số lượng layer lên tới 152 nhưng nhờ áp dụng những kỹ thuật đặc biệt mà ta sẽ tìm hiểu bên dưới nên kích thước của ResNet50 chỉ khoảng 26 triệu tham số. Kiến trúc với ít tham số nhưng hiệu quả của ResNet đã mang lại chiến thắng trong cuộc thi ImageNet năm 2015.

Những kiến trúc trước đây thường cải tiến độ chính xác nhờ gia tăng chiều sâu của mạng CNN. Nhưng thực nghiệm cho thấy đến một ngưỡng độ sâu nào đó thì độ chính xác của mô hình sẽ bão hòa và thậm chí phản tác dụng và làm cho mô hình kém chính xác hơn. Khi đi qua quá nhiều tầng độ sâu có thể làm thông tin gốc bị mất đi thì các nhà nghiên cứu của Microsoft đã giải quyết vấn đề này trên ResNet bằng cách sử dụng kết nối tắt.

Các kết nối tắt (skip connection) giúp giữ thông tin không bị mất bằng cách kết nối từ layer sớm trước đó tới layer phía sau và bỏ qua một vài layers trung gian. Trong các kiến trúc base network CNN của các mạng YOLOv2, YOLOv3 và gần đây là YOLOv4 bạn sẽ thường xuyên thấy các kết nối tắt được áp dụng.

ResNet có khối tích chập (Convolutional Bock, chính là Conv block trong hình) sử dụng bộ lọc kích thước 3 x 3 giống với của InceptionNet. Khối tích chập bao gồm 2 nhánh tích chập trong đó một nhánh áp dụng tích chập 1 x 1 trước khi cộng trực tiếp vào nhánh còn lại.

Khối xác định (Identity block) thì không áp dụng tích chập 1 x 1 mà cộng trực tiêp giá trị của nhánh đó vào nhánh còn lại.

![IdentityBlock](https://phamdinhkhanh.github.io/assets/images/20200531_CNNHistory/pic12.png)

#	Quá trình thực hiện 1 bài toán AI
## 1. Xác định vấn đề
Bước đầu tiên này là nơi mục tiêu được xác định. Géron đề cập đến các mục tiêu về mặt kinh doanh, nhưng điều này không thực sự cần thiết. Tuy nhiên, sự hiểu biết về cách giải pháp cuối cùng của hệ thống Machine Learning sẽ được sử dụng là rất quan trọng. Bước này cũng là nơi các kịch bản và các phương pháp giải quyết vấn đề có thể được so sánh và cần phải được thảo luận kỹ, cũng như các giả định được dự tính và mức độ cần thiết về chuyên môn của con người. Các mục kỹ thuật quan trọng khác xác định trong bước này bao gồm xác định loại vấn đề Machine Learning nào (được giám sát, không giám sát, v.v.) được áp dụng và các chỉ tiêu về hiệu suất nào có thể được chấp nhận.

##	2. Thu thập dữ liệu
Ở bước này dữ liệu sẽ là trọng tâm: xác định số lượng dữ liệu cần thiết, loại dữ liệu nào là cần thiết, lấy dữ liệu ở đâu, đánh giá các vấn đề pháp lý xung quanh việc thu thập dữ liệu … và tiến hành lấy dữ liệu. Một khi bạn có dữ liệu, hãy đảm bảo dữ liệu được ẩn danh một cách thích hợp, đảm bảo bạn biết loại dữ liệu đó thực sự là gì (chuỗi thời gian, quan sát, hình ảnh, v.v.), chuyển đổi dữ liệu sang định dạng bạn yêu cầu và tạo ra các tập dự liệu đào tạo, xác nhận và test.

##	3. Mô hình hóa dữ liệu
Bước này là lúc để mô hình hóa dữ liệu và thu nhỏ bộ mô hình ban đầu xuống thành một phiên bản có tiềm năng nhất. (Điều này tương tự như bước lập mô hình đầu tiên trong quy trình của Chollet: mô hình tốt → mô hình “quá tốt”, bạn có thể đọc thêm về đây) Những nỗ lực như vậy có thể liên quan đến việc sử dụng các mẫu của bộ dữ liệu đầy đủ để tạo điều kiện cho thời gian đào tạo cho các mô hình sơ bộ, mô hình nên cắt ngang một phạm vi rộng của các loại (cây, mạng lưới thần kinh, tuyến tính, v.v.). Các mô hình nên được xây dựng, đo lường và so sánh với nhau và các loại lỗi gây ra cho mỗi mô hình nên được nghiên cứu, cũng như các tính năng quan trọng nhất cho mỗi thuật toán được sử dụng. Các mô hình hoạt động tốt nhất nên được đưa vào danh sách rút gọn, sau đó có thể được tinh chỉnh sau đó.

##	4. Tinh chỉnh mô hình
Các mô hình được liệt kê trong danh sách ngắn bây giờ sẽ được tinh chỉnh các siêu đường kính của chúng và các phương pháp tập hợp nên được nghiên cứu ở giai đoạn này. Bộ dữ liệu đầy đủ nên được sử dụng trong bước này, nên lấy mẫu dữ liệu đã được sử dụng trong giai đoạn lập mô hình trước đó; không nên chọn mô hình tinh chỉnh nào là “người chiến thắng” mà không phải tiếp xúc với tất cả dữ liệu đào tạo hoặc so sánh với các mô hình khác cũng đã được tiếp xúc với tất cả dữ liệu đào tạo. Ngoài ra, bạn đã không phù hợp, phải không?

##	5. Khởi chạy mô hình

Chuẩn bị cho hệ thống Machine Learning sẵn sàng cho môi trường production; Nó sẽ cần phải cắm được vào một số hệ thống production hoặc chiến lược rộng hơn. Là một giải pháp phần mềm, nó sẽ được thực hiện unit test trước và cần được theo dõi đầy đủ sau khi chạy thực tế. Đào tạo lại các mô hình trên dữ liệu mới hoặc dữ liệu thay thế là một phần của bước này và nên được tính đến ở đây, ngay cả khi việc này đã được đưa ra trong các bước trước.





