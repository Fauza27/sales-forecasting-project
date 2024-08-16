import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
from datetime import timedelta

# Load dataset
data_url = 'supermarket_sales.csv'
data = pd.read_csv(data_url)

#Cleaning data
data['Date'] = pd.to_datetime(data['Date'])
data['Time'] = pd.to_datetime(data['Time'], format='%H:%M').dt.time

# Memuat model dari file pickle
with open('linear_regression_model.pkl', 'rb') as file:
    loaded_model = pickle.load(file)


# Sidebar untuk memilih jenis analisis
st.sidebar.title('Sales Forecasting')
analysis_type = st.sidebar.selectbox('Choose Analysis', 
                                     ['Sales Analysis', 'Dashboard', 'Forecasting'])


# Latar Belakang
if analysis_type == 'Sales Analysis':

    st.title("Sales Forecasting Analysis")
    st.markdown("""
        ## Latar Belakang

        ### 1. Pendahuluan
        Dalam dunia bisnis yang kompetitif saat ini, kemampuan untuk memprediksi penjualan di masa depan merupakan kunci untuk mengambil keputusan strategis. Penjualan merupakan indikator utama kesehatan bisnis dan sering kali menjadi dasar perencanaan anggaran, alokasi sumber daya, dan strategi pemasaran. Sebuah perusahaan yang mampu memperkirakan penjualan secara akurat dapat mengoptimalkan inventaris, mengurangi biaya penyimpanan, dan merespons perubahan permintaan pasar dengan lebih efektif.

        Namun, membuat prediksi penjualan bukanlah tugas yang mudah. Banyak faktor yang mempengaruhi penjualan, termasuk tren musiman, perilaku konsumen, dan kondisi ekonomi. Kesalahan dalam memprediksi penjualan bisa berujung pada kelebihan atau kekurangan stok, yang masing-masing membawa dampak negatif bagi bisnis. Oleh karena itu, diperlukan metode yang canggih dan andal untuk membantu perusahaan dalam membuat prediksi penjualan yang akurat.

        ### 2. Pentingnya Sales Forecasting
        Sales forecasting penting karena beberapa alasan utama:

        - Perencanaan Produksi: Dengan prediksi yang akurat, perusahaan dapat merencanakan produksi dengan tepat, menghindari kelebihan atau kekurangan stok.
        - Pengelolaan Persediaan: Forecasting membantu mengurangi biaya penyimpanan dan mencegah kerugian akibat produk yang tidak terjual.
        - Strategi Penjualan: Memprediksi penjualan memungkinkan perusahaan untuk merancang strategi penjualan yang lebih efektif, seperti menentukan diskon atau promosi pada waktu yang tepat.
        - Keputusan Keuangan: Forecasting yang baik membantu dalam perencanaan anggaran dan alokasi sumber daya, sehingga meningkatkan efisiensi keuangan.

        ### 3. Mengapa Menggunakan Data Science untuk Sales Forecasting
        Pendekatan tradisional dalam forecasting sering kali mengandalkan metode manual atau intuisi, yang rentan terhadap kesalahan dan bias. Dengan perkembangan dalam data science, sekarang kita dapat menggunakan teknik machine learning untuk melakukan sales forecasting dengan lebih akurat dan efisien. Keuntungan menggunakan data science dalam sales forecasting antara lain:

        - Akurasi yang Lebih Tinggi: Model machine learning dapat menganalisis pola dari data historis dan membuat prediksi yang lebih akurat dibandingkan dengan metode konvensional.
        - Penyesuaian Terhadap Perubahan Pasar: Algoritma machine learning dapat dilatih ulang secara berkala untuk menyesuaikan dengan tren pasar yang berubah, memastikan prediksi tetap relevan.
        - Identifikasi Faktor-Faktor yang Mempengaruhi Penjualan: Data science memungkinkan analisis mendalam untuk mengidentifikasi variabel-variabel yang paling mempengaruhi penjualan, seperti musim, kampanye pemasaran, atau tren ekonomi.
        - Efisiensi Waktu: Automasi dalam proses forecasting memungkinkan perusahaan untuk menghemat waktu dan fokus pada implementasi strategi bisnis.

        ### 4. Metodologi
        Proyek sales forecasting ini akan menggunakan metodologi berbasis data science dan machine learning, yang melibatkan beberapa tahapan berikut:

        - Pengumpulan Data: Mengumpulkan data historis penjualan yang mencakup berbagai variabel seperti waktu, produk, harga, dan kampanye pemasaran.
        - Eksplorasi dan Preprocessing Data: Mengeksplorasi data untuk memahami pola dan anomali, serta melakukan preprocessing seperti pengisian nilai yang hilang, normalisasi, dan encoding variabel kategorikal.
        - Pemilihan Fitur: Mengidentifikasi dan memilih fitur-fitur yang paling relevan untuk digunakan dalam model prediksi, berdasarkan korelasi dengan target penjualan.
        - Pengembangan Model: Menggunakan algoritma regresi seperti linear regression, atau Arima untuk membangun model prediksi penjualan.
        - Evaluasi Model: Mengevaluasi kinerja model menggunakan metrik seperti mean absolute error (MAE), root mean square error (RMSE), dan R-squared untuk memastikan akurasi prediksi.
        - Implementasi dan Pemantauan: Mengimplementasikan model dalam sistem operasional perusahaan dan memantau kinerjanya secara berkelanjutan, serta melakukan penyesuaian jika diperlukan.

        ### 5. Kesimpulan
        Sales forecasting adalah langkah penting dalam strategi bisnis yang dapat memberikan perusahaan keunggulan kompetitif. Dengan memanfaatkan data science dan teknik machine learning, perusahaan dapat meningkatkan akurasi prediksi penjualan, mengoptimalkan operasional, dan membuat keputusan strategis yang lebih baik. Proyek ini bertujuan untuk mengembangkan model sales forecasting yang andal dan dapat diterapkan secara efektif dalam berbagai konteks bisnis.
    
                """)
    

    #Business Undersatnding
    st.header('Business Understanding')
    st.write("""
        Berdasarkan latar belakang yang telah diuraikan, muncul pertanyaan penting: bagaimana perusahaan dapat secara efektif memprediksi penjualan di masa depan dan mengambil langkah-langkah strategis untuk mengoptimalkan operasional? Dengan memahami faktor-faktor yang mempengaruhi penjualan, perusahaan dapat merancang strategi yang lebih baik untuk mengelola inventaris, merencanakan produksi, dan meningkatkan efisiensi dalam pengelolaan sumber daya. Kemampuan untuk memprediksi penjualan dengan akurat akan memastikan perusahaan tetap kompetitif dan mampu merespons perubahan pasar dengan lebih cepat.
             """)
    
    #Problem Statement
    st.header('Problem Statements')
    st.write("""
        Untuk mencapai tujuan ini, proyek sales forecasting ini akan fokus pada menjawab beberapa pertanyaan kunci berikut:

        1. Bagaimana distribusi penjualan di berbagai cabang dan kota?
        Mengetahui distribusi penjualan di berbagai lokasi adalah langkah awal untuk memahami performa cabang dan mengidentifikasi cabang yang memerlukan perhatian lebih.

        2. Apakah terdapat perbedaan penjualan antara tipe pelanggan yang berbeda (Customer type)?
        Menganalisis kontribusi dari pelanggan baru dan loyal dapat membantu dalam merancang strategi pemasaran yang lebih tepat sasaran.

        3. Bagaimana pola pembelian berdasarkan gender?
        Menganalisis perbedaan pembelian antara pelanggan pria dan wanita dapat memberikan wawasan tentang preferensi masing-masing kelompok, yang penting untuk strategi pemasaran yang efektif.

        4. Produk mana yang paling laris?
        Mengevaluasi kinerja setiap garis produk akan membantu perusahaan dalam mengelola portofolio produk dan menentukan produk mana yang perlu dipromosikan lebih lanjut.

        5. Metode pembayaran mana yang paling sering digunakan?
        Mengetahui preferensi metode pembayaran dapat membantu dalam penawaran promosi atau program loyalitas yang sesuai.

        6. Bagaimana distribusi penjualan berdasarkan waktu (tanggal dan jam)?
        Memahami kapan penjualan puncak terjadi dapat membantu dalam perencanaan stok dan staf yang lebih efisien.

        7. Bagaimana hubungan antara unit price dan quantity?
        Analisis ini dapat memberikan wawasan apakah harga produk mempengaruhi jumlah pembelian, yang penting untuk pengelolaan harga dan promosi.

        8. Bagaimana pendapatan kotor dan margin laba berhubungan dengan total penjualan?
        Mengetahui hubungan ini dapat membantu perusahaan dalam menentukan strategi penetapan harga dan optimasi margin.

        9. Apakah rating pelanggan berhubungan dengan total penjualan?
        Menganalisis hubungan antara rating pelanggan dan penjualan dapat membantu perusahaan memahami kepuasan pelanggan dan dampaknya terhadap performa bisnis.
             """)

    #Goals
    st.header('Goals')
    st.write("""
        Untuk menjawab pertanyaan-pertanyaan di atas, akan dilakukan analisis mendalam dengan tujuan sebagai berikut:

        1. Mengembangkan Model Prediksi Penjualan: Menggunakan teknik machine learning untuk membangun model yang dapat memprediksi penjualan masa depan, berdasarkan data historis dan faktor-faktor relevan yang telah diidentifikasi.
        2. Mengidentifikasi Faktor-Faktor yang Mempengaruhi Penjualan: Menentukan faktor-faktor kunci yang berkontribusi terhadap penjualan, sehingga perusahaan dapat fokus pada aspek-aspek yang paling berdampak terhadap pendapatan.
        3. Merancang Strategi Bisnis yang Lebih Efektif: Menggunakan hasil analisis untuk mengembangkan strategi penjualan, pemasaran, dan inventaris yang lebih efektif, termasuk promosi produk yang tepat waktu dan penyesuaian harga.
        4. Meningkatkan Efisiensi Operasional: Mengoptimalkan alokasi sumber daya dengan fokus pada waktu dan produk yang paling menguntungkan, sehingga perusahaan dapat meningkatkan efisiensi dan menurunkan biaya operasional.
        
        Dengan pendekatan ini, proyek sales forecasting bertujuan untuk memberikan perusahaan alat yang kuat untuk mengelola penjualan, meningkatkan profitabilitas, dan memperkuat posisi kompetitif di pasar.
             """)

    #Informasi Dataset

    # Data Collection
    st.title("Dataset Infomation")
    st.write("Saya menggunakan dataset yang diambil dari platform Kaggle.")
    st.write("Link Dataset: [Super Market Sales Dataset](https://www.kaggle.com/datasets/arunjangir245/super-market-sales/data)")

    st.write(data.info())

    # Display data
    st.write("Preview Dataset")
    st.write(data.head())

    #Cheking data type
    st.write(data.info())
    st.write("""
             Mengecek data type pada masing masing fitur
             """)
    st.code("""
            <class 'pandas.core.frame.DataFrame'>
            RangeIndex: 1000 entries, 0 to 999
            Data columns (total 17 columns):
            #   Column                   Non-Null Count  Dtype  
            ---  ------                   --------------  -----  
            0   Invoice ID               1000 non-null   object 
            1   Branch                   1000 non-null   object 
            2   City                     1000 non-null   object 
            3   Customer type            1000 non-null   object 
            4   Gender                   1000 non-null   object 
            5   Product line             1000 non-null   object 
            6   Unit price               1000 non-null   float64
            7   Quantity                 1000 non-null   int64  
            8   Tax 5%                   1000 non-null   float64
            9   Total                    1000 non-null   float64
            10  Date                     1000 non-null   object 
            11  Time                     1000 non-null   object 
            12  Payment                  1000 non-null   object 
            13  cogs                     1000 non-null   float64
            14  gross margin percentage  1000 non-null   float64
            15  gross income             1000 non-null   float64
            16  Rating                   1000 non-null   float64
            dtypes: float64(7), int64(1), object(9)
            memory usage: 132.9+ KB
            """)
    st.write("""
             Pada dataframe data memiliki 17 feature atau column dan 1000 nilai non-null dalam setiap feature-feature yang ada. feature yang memiliki tipe data int64 sebanyak 1 sedangkan dengan feature yang memiliki tipe data float64 sebanyak 7. feature Date dan Time memiliki tipe data yang salah sehingga harus diubah, dan sisanya merupakan feature yang bertipe data kategorikal.
            """)
    st.write("""
             Mengecek Missing Value pada dataset
             """)
    st.code("""
            data.isnull().sum()
            """, language='python')
    st.write("Jumlah Missing value data pada dataframe:", data.isnull().sum())
    st.write("""
             Mengecek data duplicate pada dataset
             """)
    st.code("""
            data.duplicated().sum()
            """, language='python')
    st.write("Jumlah duplikasi data pada dataframe:", data.duplicated().sum())
    st.write("""
             Melihat deskripsi statistik pada dataset
             """)
    st.write(data.describe(include='all'))

    # Fixing Data Types
    st.title("Cleaning Data")
    st.write("Mengubah tipe data yang salah")
    st.code("""
            data['Date'] = pd.to_datetime(data['Date'])
            data['Time'] = pd.to_datetime(data['Time'], format='%H:%M').dt.time
            """, language='python')
    st.write("""
             Mengecek kembali missing value
             """)
    st.code("""
            data.isnull().sum()
            """, language='python')
    st.write("Jumlah Missing value data pada dataframe:", data.isnull().sum())
    

    # Exploratory Data Analysis
    st.title("Exploratory Data Analysis (EDA)")

    # Analysis by Branch and City
    st.subheader("Analisis Penjualan Berdasarkan Cabang dan Kota")
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(data=data, x='Branch', y='Total', hue='City', estimator=sum, errorbar=None, ax=ax)
    st.pyplot(fig)
    st.write("""
            - Grafik ini menunjukkan total penjualan berdasarkan cabang dan kota.
            - Cabang C (Naypyitaw) memiliki penjualan tertinggi, diikuti oleh Cabang A (Yangon) dan B (Mandalay).
            - Tidak ada perbedaan signifikan dalam penjualan antara cabang A dan B, tetapi cabang C menunjukkan sedikit keunggulan dalam total penjualan.
             """)

    # Analysis by Customer Type
    st.subheader("Analisis Tipe Pelanggan")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.barplot(data=data, x='Customer type', y='Total', estimator=sum, errorbar=None, ax=ax)
    st.pyplot(fig)
    st.write("""
            - Grafik ini menunjukkan total penjualan berdasarkan tipe pelanggan, yaitu Member dan Normal.
            - Penjualan dari pelanggan Member dan Normal terlihat sangat seimbang, dengan kontribusi yang hampir sama terhadap total penjualan.
            - Hal ini mengindikasikan bahwa baik pelanggan baru maupun pelanggan yang sudah terdaftar sebagai member memberikan kontribusi yang setara dalam penjualan.

             """)

    # Analysis by Gender
    st.subheader("Analisis Gender")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.boxplot(data=data, x='Gender', y='Total', ax=ax)
    st.pyplot(fig)
    st.write("""
            - Grafik ini adalah box plot yang menunjukkan distribusi total penjualan berdasarkan gender.
            - Median penjualan untuk pelanggan pria dan wanita hampir sama, dengan rentang distribusi yang juga serupa.
            - Tidak ada perbedaan yang signifikan antara total penjualan yang dilakukan oleh pelanggan pria dan wanita. Kedua gender menunjukkan distribusi penjualan yang mirip.
             """)

    # Analysis by Product Line
    st.subheader("Analisis Garis Produk")
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(data=data, x='Product line', y='Total', estimator=sum, errorbar=None, ax=ax)
    st.pyplot(fig)
    st.write("""
            - Grafik ini menunjukkan total penjualan per garis produk.
            - Semua garis produk, termasuk Health and Beauty, Electronic Accessories, Home and Lifestyle, Sports and Travel, Food and Beverages, dan Fashion Accessories, memiliki total penjualan yang relatif seimbang.
            - Tidak ada garis produk yang mendominasi penjualan secara signifikan, menunjukkan distribusi yang cukup merata di antara produk yang ditawarkan
             """)

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.scatterplot(data=data, x='Unit price', y='Quantity', hue='Product line', ax=ax)
    st.pyplot(fig)
    st.write("""
            - Grafik ini adalah scatter plot yang menunjukkan hubungan antara harga satuan (unit price) dan jumlah pembelian (quantity) per garis produk.
            - Garis produk ditandai dengan warna yang berbeda, namun tidak ada pola jelas yang menunjukkan bahwa harga yang lebih tinggi cenderung dibeli dalam jumlah yang lebih kecil atau sebaliknya.
            - Distribusi unit price dan quantity tersebar secara merata di berbagai garis produk, menunjukkan variasi dalam preferensi pelanggan terkait harga dan kuantitas pembelian.
             """)

    # Analysis by Payment Method
    st.subheader("Analisis Metode Pembayaran")
    fig, ax = plt.subplots(figsize=(8, 6))
    data['Payment'].value_counts().plot.pie(autopct='%1.1f%%', colors=sns.color_palette("Set2"), ax=ax)
    st.pyplot(fig)
    st.write("""
            - Pie chart ini menunjukkan distribusi penggunaan metode pembayaran di antara pelanggan.
            - Ewallet merupakan metode pembayaran yang paling sering digunakan dengan persentase 34,5%, diikuti oleh Cash dengan 34,4%, dan Credit Card dengan 31,1%.
            - Distribusi ini relatif merata, meskipun Ewallet sedikit lebih dominan dibanding metode pembayaran lainnya.
             """)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.boxplot(data=data, x='Payment', y='Total', ax=ax)
    st.pyplot(fig)
    st.write("""
            - Box plot ini membandingkan total penjualan yang dihasilkan dari masing-masing metode pembayaran.
            - Median penjualan untuk ketiga metode pembayaran cukup seimbang, dengan distribusi penjualan yang mirip.
            - Ada beberapa outliers yang menunjukkan adanya transaksi dengan total penjualan yang sangat tinggi pada semua metode pembayaran.
             """)

    # Analysis by Sales Time
    st.subheader("Analisis Waktu Penjualan")
    data['Hour'] = data['Time'].apply(lambda x: x.hour)

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.lineplot(data=data, x='Date', y='Total', estimator=sum, ax=ax)
    st.pyplot(fig)
    st.write("""
            - Line chart ini menunjukkan tren penjualan harian selama periode waktu tertentu.
            - Penjualan cenderung fluktuatif setiap hari, dengan beberapa puncak penjualan pada tanggal-tanggal tertentu.
            - Tidak ada pola yang jelas seperti peningkatan atau penurunan yang konsisten, tetapi tren umum menunjukkan variasi penjualan yang cukup tinggi setiap hari.
             """)

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.heatmap(data.pivot_table(values='Total', index='Hour', columns='Date', aggfunc='sum'), cmap='YlGnBu', ax=ax)
    st.pyplot(fig)
    st.write("""
            - Heatmap ini menunjukkan intensitas penjualan berdasarkan kombinasi waktu (jam) dan tanggal.
            - Warna yang lebih gelap menunjukkan volume penjualan yang lebih tinggi.
            - Penjualan cenderung lebih tinggi di beberapa jam tertentu pada beberapa tanggal, menunjukkan waktu-waktu puncak yang mungkin terkait dengan kebiasaan pelanggan.
             """)

    # Analysis by Unit Price and Quantity
    st.subheader("Analisis Harga Satuan dan Jumlah Pembelian")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(data=data, x='Unit price', y='Quantity', ax=ax)
    st.pyplot(fig)
    st.write("""
            - Scatter plot ini menunjukkan hubungan antara harga satuan (unit price) dan jumlah pembelian (quantity).
            - Tidak ada pola yang jelas di antara dua variabel ini, yang menunjukkan bahwa produk dengan harga tinggi atau rendah dapat dibeli dalam jumlah besar atau kecil tanpa ada kecenderungan khusus.
             """)

    # Analysis by Gross Income and Margin
    st.subheader("Analisis Pendapatan Kotor dan Margin Laba")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(data=data, x='gross income', y='gross margin percentage', hue='Product line', ax=ax)
    st.pyplot(fig)
    st.write("""
            - Scatter plot ini menunjukkan hubungan antara gross income, gross margin percentage, dan garis produk.
            - Gross margin percentage tampak relatif konstan di seluruh garis produk, dengan nilai sekitar 4,76%.
            - Gross income bervariasi tergantung pada garis produk, tetapi tidak ada hubungan signifikan yang terlihat antara gross margin dan gross income dalam data ini.
             """)

    # Analysis by Customer Rating
    st.subheader("Analisis Rating Pelanggan")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(data=data, x='Rating', y='Total', ax=ax)
    st.pyplot(fig)
    st.write("""
            - Distribusi Rating: Rating pelanggan berkisar antara 4 hingga 10, dengan titik-titik yang tersebar merata di seluruh rentang ini.
            - Distribusi Penjualan: Total penjualan bervariasi dari 0 hingga lebih dari 1000, tanpa ada tren yang jelas yang menunjukkan hubungan kuat antara rating dan total penjualan.
            - Ketiadaan Hubungan yang Kuat: Dari pola scatter plot ini, tidak tampak ada hubungan linear atau tren yang jelas antara rating dan total penjualan. Penjualan tinggi dapat terjadi pada rating rendah atau tinggi, dan sebaliknya.
            - Kepadatan: Terdapat kepadatan yang lebih tinggi di area penjualan yang lebih rendah (di bawah 400) yang tersebar di berbagai rating. Hal ini menunjukkan bahwa sebagian besar transaksi terjadi dengan total penjualan yang relatif kecil, terlepas dari rating yang diberikan.
            Secara keseluruhan, gambar ini menunjukkan bahwa tidak ada hubungan yang signifikan atau pola yang jelas antara rating yang diberikan oleh pelanggan dan total penjualan. Hal ini bisa menunjukkan bahwa faktor-faktor lain selain rating mungkin lebih mempengaruhi jumlah total penjualan.
             """)
    
    # Kesimpulan
    st.title('Kesimpulan')
    st.markdown("""
            1. Analisis Penjualan Berdasarkan Cabang dan Kota:
            Distribusi Penjualan:
            Distribusi penjualan relatif merata di antara cabang-cabang yang berbeda (A, B, C) dan kota-kota yang terlibat (Yangon, Naypyitaw, Mandalay).
            Tidak ada perbedaan signifikan dalam total penjualan antara cabang yang satu dengan yang lainnya. Ketiga cabang memiliki performa penjualan yang hampir setara.
            - Kesimpulan:
            Cabang-cabang di berbagai kota memiliki kinerja yang hampir seragam, menunjukkan bahwa tidak ada satu cabang atau kota yang mendominasi dalam hal penjualan.

            2. Analisis Tipe Pelanggan:
            Distribusi Penjualan Berdasarkan Tipe Pelanggan:
            Penjualan dari pelanggan tipe "Member" dan "Normal" hampir sama besar, tanpa ada perbedaan signifikan antara keduanya.
            Kedua tipe pelanggan berkontribusi hampir setara terhadap total penjualan.
            - Kesimpulan:
            Tidak ada perbedaan yang mencolok antara pelanggan baru dan pelanggan loyal dalam hal kontribusi terhadap penjualan. Kedua tipe pelanggan memiliki potensi yang sama dalam menghasilkan pendapatan.

            3. Analisis Gender:
            Distribusi Penjualan Berdasarkan Gender:
            Tidak ada perbedaan signifikan dalam total penjualan antara pelanggan pria dan wanita.
            Pola pembelian antara pria dan wanita hampir sama, dengan distribusi penjualan yang relatif merata.
            - Kesimpulan:
            Gender tidak mempengaruhi pola pembelian secara signifikan, dengan kontribusi yang hampir setara dari pelanggan pria dan wanita.

            4. Analisis Garis Produk (Product Line):
            Kinerja Garis Produk:
            Garis produk "Health and Beauty" dan "Electronic Accessories" merupakan yang paling laris.
            Dalam hal unit price, quantity, dan total penjualan, tidak ada satu garis produk yang secara signifikan lebih unggul di semua kategori. Namun, garis produk tertentu cenderung lebih unggul dalam kategori tertentu.
            - Kesimpulan:
            "Health and Beauty" dan "Electronic Accessories" adalah garis produk yang paling menguntungkan. Setiap garis produk memiliki kekuatan dan kelemahan masing-masing dalam hal harga, kuantitas, dan penjualan.

            5. Analisis Metode Pembayaran:
            Preferensi Metode Pembayaran:
            Metode pembayaran "E-wallet" sedikit lebih populer daripada "Cash" dan "Credit Card", meskipun perbedaannya tidak terlalu signifikan.
            Tidak ada bukti yang jelas bahwa metode pembayaran tertentu lebih sering digunakan untuk pembelian dalam jumlah besar.
            - Kesimpulan:
            "E-wallet" sedikit lebih disukai oleh pelanggan, tetapi semua metode pembayaran digunakan secara hampir merata.

            6. Analisis Waktu Penjualan:
            Distribusi Penjualan Berdasarkan Waktu:
            Tren penjualan menunjukkan fluktuasi yang cukup tajam, dengan beberapa puncak penjualan yang tersebar sepanjang bulan.
            Waktu puncak penjualan cenderung terjadi pada hari-hari tertentu tanpa pola yang jelas berdasarkan jam atau tanggal.
            - Kesimpulan:
            Penjualan bervariasi sepanjang hari dan bulan, tanpa adanya waktu puncak yang konsisten.

            7. Analisis Harga Satuan dan Jumlah Pembelian:
            Hubungan Unit Price dan Quantity:
            Tidak ada hubungan yang jelas antara unit price dan quantity, menunjukkan bahwa produk dengan harga lebih tinggi tidak selalu dibeli dalam jumlah lebih kecil, dan sebaliknya.
            - Kesimpulan:
            Harga produk tidak secara signifikan mempengaruhi jumlah pembelian, dengan berbagai produk dijual dalam berbagai kuantitas terlepas dari harga satuannya.

            8. Analisis Pendapatan Kotor dan Margin Laba:
            Hubungan Gross Income, Gross Margin, dan Total Penjualan:
            Gross income menunjukkan variasi yang besar, tetapi margin laba tetap relatif konsisten di berbagai produk dan tipe pelanggan.
            Produk atau tipe pelanggan tertentu tidak menunjukkan margin laba yang lebih tinggi secara signifikan.
            - Kesimpulan:
            Margin laba cukup konsisten di seluruh produk dan tipe pelanggan, yang menunjukkan bahwa tidak ada strategi penetapan harga yang khusus untuk meningkatkan margin pada segmen tertentu.

            9. Analisis Rating Pelanggan:
            Hubungan Rating dan Total Penjualan:
            Tidak ada hubungan yang signifikan antara rating yang diberikan oleh pelanggan dan total penjualan. Penjualan tinggi dapat terjadi pada rating rendah atau tinggi, dan sebaliknya.
            - Kesimpulan:
            Rating pelanggan tidak secara signifikan mempengaruhi total penjualan, sehingga faktor lain mungkin lebih penting dalam mendorong penjualan.

            Kesimpulan Umum:
            Dari analisis di atas, terlihat bahwa distribusi penjualan relatif merata di berbagai dimensi, seperti cabang, kota, tipe pelanggan, gender, metode pembayaran, dan garis produk. Tidak ada faktor tunggal yang dominan dalam mempengaruhi total penjualan, yang menunjukkan bahwa banyak faktor kecil yang berkontribusi secara bersama-sama.
            Pola waktu penjualan menunjukkan fluktuasi yang besar, tetapi tanpa pola puncak yang jelas, yang menyiratkan bahwa faktor eksternal atau musiman mungkin mempengaruhi penjualan.
                """)
    
    #Recomendation Action
    st.title('Rekomendation Action')
    st.markdown("""
            1. Optimasi Penawaran Produk: Fokus pada produk-produk dalam kategori "Health and Beauty" dan "Electronic Accessories" yang menunjukkan kinerja kuat. Pertimbangkan untuk memperkenalkan promosi silang atau bundling dalam kategori ini.

            2. Penargetan Pelanggan: Meskipun tidak ada perbedaan signifikan antara tipe pelanggan, program loyalitas yang ditingkatkan dapat membantu mempertahankan pelanggan "Member" dan mendorong lebih banyak pembelian berulang.

            3. Diversifikasi Metode Pembayaran: Meskipun "E-wallet" sedikit lebih populer, penting untuk tetap menawarkan berbagai metode pembayaran untuk memenuhi preferensi pelanggan yang beragam.

            4. Strategi Penjualan Berdasarkan Waktu: Mengidentifikasi faktor-faktor eksternal yang mungkin mempengaruhi fluktuasi penjualan dapat membantu dalam merencanakan kampanye pemasaran dan promosi yang lebih efektif pada waktu-waktu tertentu.

            5. Peningkatan Kualitas Layanan: Meskipun rating tidak secara langsung mempengaruhi penjualan, menjaga atau meningkatkan kualitas layanan tetap penting untuk mempertahankan kepuasan pelanggan dan mencegah churn.
                """)

    
    #Feature Enginerring
    st.header("Feature Engineering")
    st.write("""
             Sekarang akan dilakukan proses pembuatan model machine learning untuk memprediksi total penjualan selama beberapa minggu kedepan
             """)
    st.code("""
        #Menambahkan fitur 'Month' dan 'Day' dari kolom 'Date'

        data['Month'] = data['Date'].dt.month
        data['Day'] = data['Date'].dt.day

        #Menambahkan fitur 'Weekday' untuk melihat hari dalam seminggu

        data['Weekday'] = data['Date'].dt.weekday

        #Membuat fitur agregasi penjualan berdasarkan 'Date' (agregasi per hari)

        daily_sales = data.groupby('Date').agg({
        'Total': 'sum',   # Total penjualan harian
        'Quantity': 'sum' # Total jumlah barang terjual harian
        }).reset_index()
            """, language='python')
    
    #Feature Preprocessing
    st.header("Feature Preprocessing")
    st.code("""
        # Mengisi missing values (jika ada)
        data = data.ffill()

        # Mengubah tipe data kategorikal menjadi numerikal (one-hot encoding)
        data = pd.get_dummies(data, columns=['Branch', 'City', 'Customer type', 'Gender', 'Product line', 'Payment'])

        # Membuat kolom target 'Sales' dari 'Total' untuk time series forecasting
        data.set_index('Date', inplace=True)

        # Menyimpan data yang telah dipreprocessing untuk model ARIMA
        preprocessed_data = daily_sales[['Date', 'Total']].copy()
        preprocessed_data.set_index('Date', inplace=True)

        preprocessed_data.head()

        weekly_data = preprocessed_data['Total'].resample('W').sum()

        data = weekly_data.copy()
            """)
    
    #Membangun Model
    st.header("Pembangunan Model")
    st.write("""
        Pada tahap ini, dua algoritma machine learning digunakan untuk meramalkan total penjulan mingguan yang akan datang, algoritma yang dipilih adalah ARIMA dan Linear Regression.
        Pemilihan ARIMA dan Linear Regression sebagai model untuk sales forecasting didasarkan pada karakteristik dan kemampuan kedua model dalam menangani data time series dan hubungan linear antara variabel-variabel.

        1. ARIMA (AutoRegressive Integrated Moving Average):
        Keunggulan: ARIMA dirancang khusus untuk data time series dan sangat efektif dalam menangani tren, musiman, dan autokorelasi dalam data historis. ARIMA menggabungkan tiga komponen utama: AutoRegressive (AR), Integrated (I), dan Moving Average (MA), yang membuatnya mampu menangkap pola dalam data time series, termasuk tren jangka panjang dan pola musiman.
        Alasan Pemilihan: ARIMA cocok digunakan untuk forecasting penjualan yang dipengaruhi oleh tren historis, fluktuasi musiman, atau pola lain yang mungkin ada dalam data time series. ARIMA efektif ketika data stasioner atau bisa dibuat stasioner dengan diferensiasi.

        2. Linear Regression:
        Keunggulan: Linear Regression adalah model yang sederhana dan mudah diinterpretasi, yang mengasumsikan adanya hubungan linear antara variabel independen (misalnya, faktor-faktor yang mempengaruhi penjualan) dan variabel dependen (penjualan). Ini dapat digunakan untuk memprediksi nilai masa depan berdasarkan hubungan historis antara variabel-variabel tersebut.
        Alasan Pemilihan: Linear Regression cocok digunakan ketika kita memiliki variabel-variabel independen tambahan yang mungkin mempengaruhi penjualan, seperti promosi, harga, atau faktor ekonomi lainnya. Model ini baik untuk menangkap pola linier yang mungkin ada di data.
        Kombinasi kedua model ini memberikan fleksibilitas, di mana ARIMA menangani pola time series yang kompleks, sementara Linear Regression dapat mengakomodasi hubungan antara variabel independen dan penjualan.
             """)
    st.subheader("ARIMA")
    st.code("""
        from statsmodels.tsa.arima.model import ARIMA
        from sklearn.metrics import mean_squared_error

        # Memisahkan data menjadi train dan test set (90% train dan 10% test)
        train_size = int(len(weekly_data) * 0.9)
        train, test = weekly_data[:train_size], weekly_data[train_size:]

        # Membuat dan melatih model ARIMA untuk prediksi mingguan
        model = ARIMA(train, order=(5, 1, 0))
        model_fit = model.fit()

        # Menampilkan ringkasan dari model
        print(model_fit.summary())

        # Melakukan prediksi pada train set untuk melihat performa model
        train_pred = model_fit.predict(start=0, end=len(train)-1, typ='levels')

        # Melakukan prediksi pada test set
        test_pred = model_fit.predict(start=len(train), end=len(weekly_data)-1, typ='levels')

        # Menghitung RMSE untuk mengukur performa prediksi
        rmse = np.sqrt(mean_squared_error(test, test_pred))
        print(f'Test RMSE: {rmse}')

        mae = mean_absolute_error(test, test_pred)
        mse = mean_squared_error(test, test_pred)
        rmse = np.sqrt(mse)

        print("Mean Absolute Error (MAE):", mae)
        print("Mean Squared Error (MSE):", mse)
        print("Root Mean Squared Error (RMSE):", rmse)
            """, language='python')
    st.code("""
        Mean Absolute Error (MAE): 5800.10888581415
        Mean Squared Error (MSE): 66886534.30923544
        Root Mean Squared Error (RMSE): 8178.418814736467
            """, language='python')
    st.subheader("Linear Regression")
    st.code("""
        from sklearn.model_selection import train_test_split
        from sklearn.linear_model import LinearRegression
        from sklearn.metrics import mean_absolute_error, mean_squared_error

        # Memisahkan data menjadi train dan test set (90% train dan 10% test)
        train_size = int(len(weekly_data) * 0.9)
        train, test = weekly_data[:train_size], weekly_data[train_size:]

        X_train = np.arange(len(train)).reshape(-1, 1)
        y_train = train.values

        X_test = np.arange(len(test)).reshape(-1, 1)
        y_test = test.values

        # Membuat dan melatih model Linear Regression
        lr_model = LinearRegression()
        lr_model.fit(X_train, y_train)

        # Melakukan prediksi pada test set
        y_pred = lr_model.predict(X_test)

        # Evaluasi model
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)

        print("Mean Absolute Error (MAE):", mae)
        print("Mean Squared Error (MSE):", mse)
        print("Root Mean Squared Error (RMSE):", rmse)
            """, language='python')
    st.code("""
        Mean Absolute Error (MAE): 3725.5272409090903
        Mean Squared Error (MSE): 22965455.30905241
        Root Mean Squared Error (RMSE): 4792.228636975954
            """, language='python')
    
    #Hasil Evaluasi untuk Logistic Regression
    st.header("Evaluation Model")
    st.write("""
        Berikut adalah hasil metrik evaluasi dari kedua model:
        
        ARIMA:
        
        MAE: 5800.11
        
        MSE: 66886534.31
        
        RMSE: 8178.42

        Linear Regression:
        
        MAE: 3725.53
        
        MSE: 22965455.31
        
        RMSE: 4792.23

        Kesimpulan: Linear Regression lebih baik daripada ARIMA karena memiliki nilai MAE, MSE, dan RMSE yang lebih rendah.
             """)
    
    st.header("Hyperparamter Tunning")
    st.write("""
    Sekarang akan dilakukan hyperparameter tuning untuk meningkatkan performa model pada Linear Regression:
    """)
    st.code("""
        from sklearn.linear_model import Ridge
        from sklearn.model_selection import GridSearchCV

        # Definisikan grid hyperparameter untuk Ridge Regression
        param_grid = {
        'alpha': [0.1, 1.0, 10.0, 100.0]
        }

        # Membuat objek model Ridge
        ridge = Ridge()

        # Membuat GridSearchCV untuk hyperparameter tuning
        grid_search = GridSearchCV(estimator=ridge, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)

        # Melatih GridSearchCV pada data training
        grid_search.fit(X_train, y_train)

        # Melihat hasil terbaik
        best_lrmodel = grid_search.best_estimator_
        print("Best hyperparameters:", grid_search.best_params_)
        print("Best model score:", grid_search.best_score_)

        # Melakukan prediksi pada test set menggunakan model terbaik
        y_pred = best_lrmodel.predict(X_test)

        # Evaluasi model
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)

        print("Mean Absolute Error (MAE):", mae)
        print("Mean Squared Error (MSE):", mse)
        print("Root Mean Squared Error (RMSE):", rmse)
            """, language='python')
    st.code("""
        Mean Absolute Error (MAE): 3675.89415
        Mean Squared Error (MSE): 25490586.1202171
        Root Mean Squared Error (RMSE): 5048.820270144017
            """, language='python')
    st.write("""
             setelah dilakukan hyperparameter tuning, model menunjukan peningkatan dengan ditandai menurunnya nilai MAE, MSE, dan RMSE, sehingga
             Model Linear Regression dakan dipilih untuk melakukan sales forecasting.
             """)



# Dashboard penjualan
if analysis_type == 'Dashboard':
    # Feature Engineering
    data['Month'] = data['Date'].dt.month
    data['Day'] = data['Date'].dt.day
    data['Weekday'] = data['Date'].dt.weekday

    sns.set(style='darkgrid')

    # Sidebar untuk filter rentang waktu
    with st.sidebar:
        st.header('Filter Data')
        start_date = st.date_input('Start date', data['Date'].min())
        end_date = st.date_input('End date', data['Date'].max())
        branch_filter = st.multiselect('Branch', data['Branch'].unique(), data['Branch'].unique())
        city_filter = st.multiselect('City', data['City'].unique(), data['City'].unique())

    filtered_data = data[
        (data['Date'] >= pd.to_datetime(start_date)) & 
        (data['Date'] <= pd.to_datetime(end_date)) &
        (data['Branch'].isin(branch_filter)) &
        (data['City'].isin(city_filter))
    ]

    st.title('Sales Dashboard')

    # Visualisasi 1: Penjualan berdasarkan tanggal (line chart)
    st.subheader('Penjualan Berdasarkan Tanggal')
    daily_sales = filtered_data.groupby('Date')['Total'].sum().reset_index()

    fig, ax = plt.subplots()
    sns.lineplot(data=daily_sales, x='Date', y='Total', ax=ax, color='green')    
    plt.title('Total Penjualan per Tanggal')
    plt.xlabel('Tanggal')
    plt.ylabel('Total Penjualan')
    plt.xticks(rotation=45)
    st.pyplot(fig)

    # Visualisasi 2: Penjualan berdasarkan bulan
    st.subheader('Penjualan Berdasarkan Bulan')
    monthly_sales = filtered_data.groupby('Month')['Total'].sum().reset_index()

    fig, ax = plt.subplots()
    sns.barplot(data=monthly_sales, x='Month', y='Total', ax=ax, palette='Blues_d')
    plt.title('Total Penjualan per Bulan')
    st.pyplot(fig)

    # Visualisasi 3: Penjualan berdasarkan Cabang
    st.subheader('Penjualan Berdasarkan Cabang')
    branch_sales = filtered_data.groupby('Branch')['Total'].sum().reset_index()

    fig, ax = plt.subplots()
    sns.barplot(data=branch_sales, x='Branch', y='Total', ax=ax, palette='Oranges_d')
    plt.title('Total Penjualan per Cabang')
    st.pyplot(fig)

    # Visualisasi 4: Distribusi Rating
    st.subheader('Distribusi Rating')
    fig, ax = plt.subplots()
    sns.histplot(filtered_data['Rating'], bins=10, kde=True, ax=ax, color='green')
    plt.title('Distribusi Rating Pelanggan')
    st.pyplot(fig)

    # Visualisasi 5: Penjualan Berdasarkan Garis Produk
    st.subheader('Penjualan Berdasarkan Garis Produk')
    product_sales = filtered_data.groupby('Product line')['Total'].sum().reset_index()

    fig, ax = plt.subplots()
    sns.barplot(data=product_sales, x='Product line', y='Total', ax=ax, palette='Purples_d')
    plt.title('Total Penjualan per Garis Produk')
    plt.xticks(rotation=45)
    st.pyplot(fig)

# Sales Forecasting
if analysis_type == 'Forecasting':
        # Feature Engineering
        data['Month'] = data['Date'].dt.month
        data['Day'] = data['Date'].dt.day
        data['Weekday'] = data['Date'].dt.weekday

        # Membuat fitur agregasi penjualan harian
        daily_sales = data.groupby('Date').agg({
        'Total': 'sum',  # Total penjualan harian
        'Quantity': 'sum'  # Total jumlah barang terjual harian
        }).reset_index()

        # Set index Date
        daily_sales.set_index('Date', inplace=True)

        # Resample data menjadi weekly sales
        weekly_sales = daily_sales['Total'].resample('W').sum()

        # Streamlit App Title
        st.title("Sales Forecasting App with Linear Regression")

        # Tampilkan beberapa baris data
        st.subheader("Dataset Supermarket Sales")
        st.write(data.head())

        # Visualisasi Data Historis Penjualan Mingguan
        st.subheader("Visualisasi Penjualan Mingguan")
        st.line_chart(weekly_sales)

        # User input: berapa minggu ke depan untuk forecasting
        weeks_ahead = st.number_input("Masukkan jumlah minggu ke depan untuk prediksi:", min_value=1, max_value=52, value=12)

        # Prediksi jika tombol ditekan
        if st.button("Prediksi Penjualan"):
                # Menghitung minggu masa depan yang akan diprediksi
                last_week_index = len(weekly_sales)
                future_weeks = np.arange(last_week_index, last_week_index + weeks_ahead).reshape(-1, 1)

                # Melakukan prediksi dengan model yang telah diload
                future_predictions = loaded_model.predict(future_weeks)

                # Membuat DataFrame untuk hasil prediksi
                future_dates = pd.date_range(start=weekly_sales.index[-1] + timedelta(weeks=1), periods=weeks_ahead, freq='W')
                future_predictions_df = pd.DataFrame(data=future_predictions, index=future_dates, columns=['Predicted Sales'])

                # Menampilkan hasil prediksi
                st.subheader(f"Prediksi penjualan untuk {weeks_ahead} minggu ke depan:")
                st.write(future_predictions_df)

                # Visualisasi hasil prediksi
                st.line_chart(future_predictions_df)

                # Plotting gabungan data historis dan prediksi masa depan
                st.subheader("Visualisasi penjualan historis dan prediksi masa depan")
                combined_data = pd.concat([weekly_sales, future_predictions_df])
                st.line_chart(combined_data)

                # Analisis tambahan: Visualisasi distribusi produk yang terjual
                st.subheader("Distribusi Produk yang Terjual per Garis Produk")
                product_sales = data.groupby('Product line')['Quantity'].sum().reset_index()
                sns.barplot(x='Product line', y='Quantity', data=product_sales)
                plt.xticks(rotation=45)
                st.pyplot(plt)





