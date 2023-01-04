# Report

## Domain Proyek

*Customer churn* adalah suatu istilah yang merujuk pada sejumlah *customer* yang tidak lagi menggunakan dan membayar produk atau layanan. Prediksi *customer churn* pada suatu bank berarti memprediksi perilaku customer pada bank apakah mengindikasikan *customer churn* atau bukan. Prediksi ini penting bagi suatu bank karena mendapatkan *customer* atau nasabah baru lebih membutuhkan biaya dibanding mempertahankannya. Selain itu juga, dapat meningkatkan pendapatan perusahaan dengan banyaknya nasabah yang bertahan pada bank maka pendapatan akan bertambah. Perilaku nasabah yang berbeda-beda membuat perusahaan sulit untuk melihat secara langsung adanya kemungkinan nasabah yang churn. Sehingga, pengidentifikasian *customer churn* dapat dilakukan dengan metode machine learning yaitu klasifikasi khususnya menggunakan klasifikasi yang *supervised learning* pada kasus ini, yaitu mengkelaskan customer yang termasuk churn atau bukan.

Setelah teridentifikasi suatu nasabah yang churn maka bank dapat memaksimalkan upaya pemasarannya agar nasabah bertahan menggunakan layanan bank misalnya dengan merujuk pada perilaku nasabah menggunakan layanan bank. Penelitian mengenai [IG-KNN untuk Prediksi Customer Churn Telekomunikasi](https://www.researchgate.net/publication/316591760_IG-KNN_UNTUK_PREDIKSI_CUSTOMER_CHURN_TELEKOMUNIKASI) telah dilakukan yaitu dengan menggunakan metode IG-KNN dan menghasilkan tingkat akurasi mencapai 89% dengan jumlah data yaitu 5000 data.


## Business Understanding

### Problem Statements

- Kesulitan perusahaan bank mengetahui perilaku nasabah yang termasuk *customer churn* secara langsung (manual) karena perbedaan perilaku setiap nasabah.

### Goals

- Memprediksi *customer churn* pada perusahaan bank berdasarkan data nasabah pada layanan bank yaitu dengan mengklasifikasikannya ke dalam dua kelas yaitu nasabah churn dan bukan churn.

    ### Solution statements
    - Melakukan proses Exploratory Data Analytics (EDA) dan preprocessing data sebelum melakukan prediksi pada data.
    - Menggunakan model machine learning dalam proses pengklasifikasian *customer churn* yaitu algoritma supervised learning, K-Nearest Neighbor dan Random Forest
    - Mengukur dan membandingkan metric klasifikasi dari kedua model untuk memilih model terbaik.

## Data Understanding
Data yang digunakan pada proyek ini adalah data perusahaan bank yaitu ABC Multistate Bank yang dapat diakses pada [Kaggle](https://www.kaggle.com/datasets/gauravtopre/bank-customer-churn-dataset?datasetId=2445309) yang terdiri dari 11 variabel.

### Variabel-variabel pada ABC Multistate Bank dataset adalah sebagai berikut:
- customer_id: nomor unik pada setiap nasabah.
- credit_score: score yang ditetapkan pada nasabah berdasarkan perilakunya dalam membayarkan credit.
- country: negara nasabah.
- gender: jenis kelamin nasabah.
- age: umur nasabah.
- tenure: lama waktu nasabah untuk mengembalikan pinjaman (credit) dan bunganya.
- balance: jumlah uang pada rekening nasabah.
- products_number: nomor produk yang digunakan nasabah.
- credit_card: kartu kredit (apakah nasabah punya kartu kredit? 0: tidak, 1: ya)
- active_member: keaktifan member nasabah (0: tidak aktif, 1: aktif)
- estimated_salary: estimasi pendapatan nasabah.
- churn: jika nasabah meninggalkan perusahaan bank pada suatu periode waktu (0: tidak, 1: ya)

Sebelum melakukan visualisasi data setiap variabel, hal pertama yang dilakukan adalah menangani *missing value* dan *outliers*.

**EDA**:
- Sebelumnya kolom customer_id dihapus terlebih dahulu karena tidak diperlukan pada proses prediksi (hanya berisi nomor unik sebagai yang membedakan nasabah dengan nasabah lain). Setelah itu mengecek data jika terdapat *missing value* pada tabel 1.

Tabel 1.
|       | Credit Score | Age          | Balance       | Estimated Salary |
|-------|--------------|--------------|---------------|------------------|
| Count | 10000.000000 | 10000.000000 | 10000.000000  | 10000.000000     |
| Mean  | 650.528800   | 38.921800    | 76485.889288  | 100090.239881    |
| Std   |    96.653299 | 10.487806    | 62397.405202  | 57510.492818     |
| Min   | 350.000000   |    18.000000 | 0.000000      | 11.580000        |
| 25%   | 584.000000   | 32.000000    |      0.000000 | 51002.110000     |
| 50%   | 652.000000   |    37.000000 | 97198.540000  | 100193.915000    |
| 75%   | 718.000000   | 44.000000    | 127644.240000 | 149388.247500    |
| Max   | 850.000000   | 92.000000    | 250898.090000 | 199992.480000    |


Karena terdapat missing value pada 'balance' maka ditangani dengan mengisinya dengan nilai rata-rata kolom.

- Menangani *outliers* yang dapat dilihat dari barplot yaitu dengan menghapus setiap barisnya. Metode yang digunakan dalam pendeteksian adanya *outliers* adalah dengan melihat data jika masuk dalam batas kuartil atas dan bawah maka tidak termasuk *outliers* begitupun sebaliknya.
- Jumlah dan persentase masing-masing fitur kategori dapat dilihat sebagai berikut.

![persentase](https://user-images.githubusercontent.com/91725987/210475930-ee326951-9405-4cf0-85f9-197b1bf51472.jpg)

- Selanjutnya persebaran data untuk fitur numerik adalah sebagai berikut.

![numfitur](https://user-images.githubusercontent.com/91725987/210475989-8fac4e4a-9f63-436e-82e3-f3370e11cf7c.jpg)

- Visualisasi data untuk melihat korelasi antara fitur kategorik dan target. Salah satu yang dapat dilihat adalah customer laki-laki lebih banyak masuk dalam kategori 'tidak churn' dibanding perempuan.

![Screenshot 2023-01-03 223143](https://user-images.githubusercontent.com/91725987/210476047-ee9afcfd-979a-4756-9830-cdcd32832747.jpg)

## Data Preparation

- Data preparation langkah pertama adalah mentransformasikan data kategorikal yaitu 'gender' dan 'country' yang masih berupa tipe objek menjadi angka biner menggunakan metode One Hot Encoding agar tidak terjadi error saat training maupun testing (mesin hanya menerima input berupa angka)
- Kedua, membagi data ke dalam training dan testing dataset untuk variabel fitur dan target dengan proporsi 80% training, 20% testing.
- Ketiga, melakukan standardisasi pada fitur numerikal agar fitur memiliki interval nilai yang lebih kecil dan seragam sehingga, mengurangi lama proses pelatihan.

## Modeling
Model yang digunakan dalam proyek ini adalah Random Forest dan K-Nearest Neighbor. Random forest adalah salah satu jenis model yang digunakan untuk kasus klasifikasi yang bekerja menggunakan *decision tree* yaitu dengan memilih parameter n_estimator = 100 dan max_depth = 5. Adapun kekurangan dari model random forest yaitu pembelajarannya dapat berjalan lambat karena bergantung pada parameter, tetapi model ini dapat bekerja dengan efisien untuk data yang besar. Sedangkan, K-Nearest Neighbor (KNN) merupakan salah satu jenis model yang digunakan untuk kasus klasifikasi yang bekerja dengan melihat kesamaan data dengan parameter yang digunakan yaitu n_neighbors = 10 yang dipilih berdasarkan grafik hasil prediksi untuk 1-50 neighbor. Salah satu kekurangan dari KNN ini adalah kurang baik digunakan untuk dataset yang besar, tetapi modelnya dapat mudah beradaptasi. 

Adapun berdasarkan pelatihan yang dilakukan, model akhir yang dipilih adalah model random forest. Selain karena nilai akurasi dan presisi yang lebih tinggi dari KNN, secara keseluruhan score dari model random forest juga lebih stabil pada kedua kelas (churn dan bukan churn)

## Evaluation
Metrik evaluasi yang digunakan bergantung pada nilai *confusion matrix* yang berisi yaitu: *True Positive (TP)* atau data positif yang diprediksi benar, *True Negative (TN)* atau data negatif yang diprediksi benar,*False Positive (FP)* atau data negatif namun diprediksi sebagai data positif, *False Negative (FN)* atau data positif namun diprediksi sebagai data negatif. Metrik evaluasi yang digunakan yaitu:
- Accuracy: keakuratan model dapat mengklasifikasikan dengan benar. Dihitung dengan: 

$$ Accuracy = {TP+TN \over {TP+TN+FP+FN}} $$
- Precision: tingkat keakuratan antara data yang diminta dengan hasil prediksi yang diberikan oleh model. Dihitung dengan: 

$$ Precision = {(TP) \over (TP + FP)} $$
- Recall:  keberhasilan model dalam menemukan kembali sebuah informasi. Dihitung dengan: 

$$ Recall = {(TP) \over (TP + FN)} $$
- F1 Score: perbandingan rata-rata presisi dan recall yang dibobotkan. Dihitung dengan: 

$$ F1 = {2*(Recall*Precision) \over (Recall+Precision)} $$

Nilai dari metric-metric kedua model tersebut yaitu sebagai berikut. Dimana model1 adalah *random forest* dan model2 adalah KNN. 

![Screenshot 2023-01-03 224445](https://user-images.githubusercontent.com/91725987/210476117-696f91ee-5f61-4973-b713-05754e0e73e3.jpg)
![Screenshot 2023-01-03 224503](https://user-images.githubusercontent.com/91725987/210476122-7398b0f5-2177-4775-929c-9b716e84332f.jpg)


Berdasarkan metric tersebut dapat dilihat bahwa model *Random Forest* lebih unggul pada nilai akurasi (0.82) dan presisi (0.84) dibandingkan dengan model KNN dengan nilai akurasi dan presisi masing-masing 0.80, tetapi KNN lebih unggul pada nilai recall yaitu mencapai 1.00. Namun dari keseluruhan metric dan kestabilannya terhadap kedua kelas, *random forest* lebih stabil sehingga model *random forest* yang dijadikan sebagai model klasifikasi *customer churn*
