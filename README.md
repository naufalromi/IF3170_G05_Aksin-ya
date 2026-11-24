# Tugas Besar 2 IF3170 Inteligensi Artifisial — Kelompok G05 (Aksin-ya)

<p align="center">
  <img src="https://c.tenor.com/dj02WgyEdtEAAAAd/tenor.gif" alt="Kelompok G05 Aksin-ya">
</p>

## Deskripsi Singkat

Repository ini berisi implementasi algoritma *supervised learning* dari awal (*from scratch*) untuk memprediksi pembatalan pemesanan hotel (kolom target `is_canceled`). Setiap model dibandingkan dengan implementasi pustaka scikit-learn.

Model yang diimplementasikan:

| Model | Implementasi *scratch* | Pembanding scikit-learn |
| --- | --- | --- |
| K-Nearest Neighbors | `src/knn.py` | `KNeighborsClassifier` |
| Naive Bayes (Gaussian + Kategorikal) | `src/NaiveBayes.py` | `GaussianNB` + `CategoricalNB` |
| Support Vector Machine | `src/svm.py` | `SVC` |
| Decision Tree Learning (C4.5) | `src/dtl/` | `DecisionTreeClassifier` |
| Logistic Regression | `src/LogRes.py` | `LogisticRegression` |

Seluruh alur eksperimen ada di notebook `docs/Tugas_Besar_2_IF3170_Inteligensi_Artifisial_2026_2027.ipynb`, mulai dari eksplorasi data, *preprocessing*, pelatihan model, evaluasi *holdout* dan *k-fold cross validation*, penyimpanan dan pemuatan model, submisi Kaggle, hingga *error analysis*.

## Struktur Repository

```
.
├── docs/
│   ├── Tugas_Besar_2_IF3170_Inteligensi_Artifisial_2026_2027.ipynb   # notebook utama
│   ├── train.csv                                                      # data latih
│   └── test.csv                                                       # data uji (submisi Kaggle)
├── src/
│   ├── knn.py
│   ├── NaiveBayes.py
│   ├── svm.py
│   ├── LogRes.py
│   └── dtl/                     # Decision Tree C4.5
│       ├── algorithms.py
│       ├── c45.py
│       └── dtl_utils.py
├── output/                      # model tersimpan (.pkl), grafik, dan animasi
├── requirements.txt
└── README.md
```

## Cara Setup

Program dikembangkan dengan Python 3.14.

1. *Clone* repository lalu masuk ke direktorinya.

   ```bash
   git clone https://github.com/naufalromi/IF3170_G05_Aksin-ya.git
   cd IF3170_G05_Aksin-ya
   ```

2. Buat dan aktifkan *virtual environment*.

   ```bash
   python -m venv .venv
   ```

   Linux/macOS:

   ```bash
   source .venv/bin/activate
   ```

   Windows (PowerShell):

   ```powershell
   .venv\Scripts\Activate.ps1
   ```

3. Pasang dependensi, beserta `jupyter` untuk menjalankan notebook.

   ```bash
   pip install -r requirements.txt
   pip install jupyter
   ```

## Cara Menjalankan Program

1. Pastikan `train.csv` dan `test.csv` berada di folder `docs/`.
2. Buka notebook utama, baik melalui Jupyter maupun VS Code.

   ```bash
   jupyter notebook docs/Tugas_Besar_2_IF3170_Inteligensi_Artifisial_2026_2027.ipynb
   ```

3. Pilih kernel dari `.venv`, lalu jalankan seluruh sel secara berurutan (**Run All**). Notebook dapat dijalankan dari *root* repository maupun dari folder `docs/`.

Hasil yang dihasilkan notebook:

- Model tersimpan (`.pkl`), grafik, dan animasi kontur Logistic Regression di folder `output/`
- File submisi Kaggle `submission.csv` di *root* repository

Pelatihan model *scratch* (terutama Logistic Regression, SVM, dan C4.5) beserta *k-fold cross validation* dapat memakan waktu beberapa menit.

## Pembagian Tugas

|  | NIM | Nama | Tugas |
| :---: | :---: | :---: | :---: |
| <a href="https://github.com/Achideon"><img src="https://github.com/Achideon.png" alt="Achideon" width="60"></a><br>[@Achideon](https://github.com/Achideon) | 13524048 | Josh Reinhart Zidik | Logistic Regression & Error Analysis |
| <a href="https://github.com/naufalromi"><img src="https://github.com/naufalromi.png" alt="naufalromi" width="60"></a><br>[@naufalromi](https://github.com/naufalromi) | 13524058 | Muhammad Naufal Romi Annafi | Preprocessing & Naive Bayes |
| <a href="https://github.com/Zekiamani1"><img src="https://github.com/Zekiamani1.png" alt="Zekiamani1" width="60"></a><br>[@Zekiamani1](https://github.com/Zekiamani1) | 13524082 | Zeki Amani | Support Vector Machine |
| <a href="https://github.com/HussainDzaki"><img src="https://github.com/HussainDzaki.png" alt="HussainDzaki" width="60"></a><br>[@HussainDzaki](https://github.com/HussainDzaki) | 13524084 | Dzaki Ahmad Al Husainy | Decision Tree Learning & Model Submission |
| <a href="https://github.com/daypft"><img src="https://github.com/daypft.png" alt="daypft" width="60"></a><br>[@daypft](https://github.com/daypft) | 13524133 | Muhammad Daffa Arrizki Yanma | Preprocessing & K-Nearest Neighbors |

<p align="center">
  <img src="https://c.tenor.com/gUJgeSXy5IQAAAAC/tenor.gif" alt="Kelompok G05 Aksin-ya" width="180">
</p>

<p align="center">
  <img src="https://c.tenor.com/n2jE1_8ijFgAAAAC/tenor.gif" alt="Thank You" width="180">
</p>
