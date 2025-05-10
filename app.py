from flask import Flask, request, render_template, session, jsonify
import pandas as pd
import numpy as np
import os
import io
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')  # Gunakan backend non-interaktif
import base64
import json
import scipy.stats as stats
from scipy.stats import t as t_dist
from scipy.optimize import minimize
from scipy.stats import kstest, chi2, t
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.stats.stattools import durbin_watson
from statsmodels.tsa.arima.model import ARIMA
import warnings

# Inisialisasi Flask
app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Dibutuhkan untuk session

UPLOAD_FOLDER = 'uploads'
RETURN_FILE = os.path.join(UPLOAD_FOLDER, 'return_processed_data.csv')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

from flask import Flask, request, render_template, session, redirect
import pandas as pd
import numpy as np
import os
import io
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')  # Gunakan backend non-interaktif
import base64
import json
from statsmodels.tsa.stattools import adfuller

# Inisialisasi Flask
app = Flask(__name__)
app.secret_key = 'your_secret_key'
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/glosarium')
def glosarium():
    return render_template('glosarium.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        if not file:
            return render_template('upload.html', error="Tidak ada file yang diunggah.", current_page='upload')

        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(file_path)

        try:
            data = pd.read_csv(file_path)
            top_data = data.head(5).to_dict(orient='records')  # 5 data teratas
            bottom_data = data.tail(5).to_dict(orient='records')  # 5 data terbawah
            return render_template('upload.html', top_data=top_data, bottom_data=bottom_data, file_name=file.filename, current_page='upload')
        except Exception as e:
            return render_template('upload.html', error=f"Terjadi kesalahan saat membaca file: {e}", current_page='upload')

    return render_template('upload.html', current_page='upload')

@app.route('/prepro', methods=['GET', 'POST'])
def preprocess_data():
    if request.method == 'POST':
        # Pastikan file_name ada dalam request.form
        file_name = request.form.get('file_name')
        
        if not file_name:
            return render_template('prepro.html', error="Nama file diperlukan.", img_harga=None, current_page='prepro')

        file_path = os.path.join(UPLOAD_FOLDER, file_name)

        plot_url = None  # Inisialisasi plot_url
        try:
            # Load file
            dataload = pd.read_csv(file_path)
            dataload['Date'] = pd.to_datetime(dataload['Date'])
            dataload = dataload.sort_values(by='Date', ignore_index=True)

            # Cek missing values & duplikat
            null_counts = dataload.isnull().sum().to_string()
            duplicate_counts = dataload.duplicated().sum()

            # Mengolah data harga
            dataload['Price'] = dataload['Price'].astype(str).str.replace(',', '').astype(float)
            dataload = dataload[['Date', 'Price']]

            # Buat grafik harga saham
            plt.figure(figsize=(7, 3))
            plt.plot(dataload['Date'], dataload['Price'], label='Harga Historis', color='blue')

            # Menambahkan judul dan label
            plt.title('Grafik Harga Saham Close Price', fontsize=10)
            plt.xlabel('Date', fontsize=8)
            plt.ylabel('Harga Saham', fontsize=8)
            plt.legend(fontsize=8)
            plt.grid(True)

            # Simpan grafik ke dalam buffer
            img = io.BytesIO()
            plt.savefig(img, format='png', dpi=150, bbox_inches='tight')  # Simpan gambar ke buffer
            img.seek(0)
            plot_url = base64.b64encode(img.getvalue()).decode()
            plt.close()  # Hindari tampilan ganda

            # Simpan ke session (untuk penggunaan cepat) dan file CSV
            session['dataload'] = dataload.to_json(orient='records')
            processed_path = os.path.join(UPLOAD_FOLDER, 'processed_data.csv')
            dataload.to_csv(processed_path, index=False)

            # Statistik deskriptif
            data_description = dataload.describe().to_string()

            # Ambil 5 data teratas dan 5 data terbawah
            top_5_data = dataload.head(5).to_dict(orient='records')
            bottom_5_data = dataload.tail(5).to_dict(orient='records')

            return render_template('prepro.html',
                                   null_counts=null_counts,
                                   duplicate_counts=duplicate_counts,
                                   data_description=data_description,
                                   processed_data=dataload.head(15).to_dict(orient='records'),  # Menampilkan 15 baris teratas
                                   top_5_data=top_5_data,  # Data teratas
                                   bottom_5_data=bottom_5_data,  # Data terbawah
                                   img_harga=plot_url,
                                   current_page='prepro')  # Pass the plot URL to template

        except Exception as e:
            return render_template('prepro.html', error=f"Terjadi kesalahan saat preprocessing data: {e}", img_harga=plot_url, current_page='prepro')
    
    # Jika metode GET, hanya render template tanpa melakukan pemrosesan
    return render_template('prepro.html', img_harga=None, current_page='prepro')

@app.route('/return', methods=['GET', 'POST'])
def calculate_return():
    if 'dataload' in session:
        dataload = pd.read_json(session['dataload'])
    else:
        processed_path = os.path.join(UPLOAD_FOLDER, 'processed_data.csv')
        if os.path.exists(processed_path):
            dataload = pd.read_csv(processed_path)
        else:
            return "Data tidak ditemukan. Silakan lakukan preprocessing terlebih dahulu.", 400

    # Hitung return logaritmik
    dataload['Price'] = pd.to_numeric(dataload['Price'])  # Pastikan kolom Price numeric
    dataload['Return'] = np.log(dataload['Price'] / dataload['Price'].shift(1))

    # Hapus nilai NaN
    dataload = dataload.dropna()

    # **Simpan kembali hasil perhitungan ke session**
    session['dataload'] = dataload.to_json()
    session.modified = True

    # Buat grafik return
    img_return = create_return_plot(dataload)

    # Ambil 15 baris teratas
    dataload_top15 = dataload.head(15)

    # Simpan data yang sudah dihitung return-nya ke session
    session['dataload'] = dataload.to_json()
    # Atau simpan ke file CSV untuk diakses di halaman lain
    dataload.to_csv(os.path.join(UPLOAD_FOLDER, 'return_processed_data.csv'), index=False)

    return render_template('return.html', data=dataload_top15.to_dict(orient='records'), img_return=img_return, current_page='return')

def create_return_plot(dataload):
    plt.figure(figsize=(16, 5))
    plt.plot(dataload.index, dataload['Return'], label="Return", color='blue')
    plt.legend(loc='best')
    plt.title("Return Data")
    plt.xlabel('Index')
    plt.ylabel('Return')
    plt.grid(True)

    img = io.BytesIO()
    plt.savefig(img, format='png', dpi=150, bbox_inches='tight')  # Simpan gambar ke buffer
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    plt.close()  # Hindari tampilan ganda
    return plot_url

@app.route('/stasioneritas', methods=['GET', 'POST'])
def stasioneritas():
    if request.method == 'POST':
        # 🔹 Cek apakah session menyimpan data
        if 'dataload' in session:
            dataload = pd.read_json(session['dataload']).copy()
            print("Data dari session:", dataload.head())  # Debugging
        else:
            # 🔹 Coba ambil dari CSV jika session tidak ada
            processed_path = os.path.join(UPLOAD_FOLDER, 'return_processed_data.csv')
            if os.path.exists(processed_path):
                dataload = pd.read_csv(processed_path)
                print("Data dari CSV:", dataload.head())  # Debugging
            else:
                return render_template('stasioneritas.html', error="Data tidak ditemukan. Silakan lakukan preprocessing terlebih dahulu.", current_page='stasioneritas')

        # 🔹 Pastikan kolom 'Return' ada
        if 'Return' not in dataload.columns:
            return render_template('stasioneritas.html', error="Kolom 'Return' belum dihitung. Silakan hitung return terlebih dahulu.", current_page='stasioneritas')

        # 🔹 Jalankan ADF Test
        def adf_test(timeseries):
            dftest = adfuller(timeseries, autolag='AIC')
            return {
                'Test Statistic': dftest[0],
                'p-value': dftest[1],
                '#Lags Used': dftest[2],
                'Number of Observations Used': dftest[3],
                'Critical Value (1%)': dftest[4]['1%'],
                'Critical Value (5%)': dftest[4]['5%'],
                'Critical Value (10%)': dftest[4]['10%']
            }

        adf_results = adf_test(dataload['Return'])

        return render_template('stasioneritas.html', adf_output=adf_results, current_page='stasioneritas')

    return render_template('stasioneritas.html', current_page='stasioneritas')

@app.route('/split_data', methods=['GET', 'POST'])
def split_data():
    if not os.path.exists(RETURN_FILE):
        return render_template('split_data.html', error="File return_processed_data.csv tidak ditemukan.", current_page='split_data')

    data = pd.read_csv(RETURN_FILE)

    # Ambil dari session
    in_sample_head = session.get('in_sample_head', None)
    out_sample_head = session.get('out_sample_head', None)
    in_sample_tail = session.get('in_sample_tail', None)
    out_sample_tail = session.get('out_sample_tail', None)
    in_sample_shape = session.get('in_sample_shape', None)
    out_sample_shape = session.get('out_sample_shape', None)

    if request.method == 'POST':
        try:
            # Ambil input tanggal dari form
            in_sample_start = request.form.get('in_sample_start')
            in_sample_end = request.form.get('in_sample_end')
            out_sample_start = request.form.get('out_sample_start')
            out_sample_end = request.form.get('out_sample_end')

            if not all([in_sample_start, in_sample_end, out_sample_start, out_sample_end]):
                return render_template('split_data.html', error="Semua kolom harus diisi.", current_page='split_data')

            # Konversi kolom Date ke datetime
            data['Date'] = pd.to_datetime(data['Date'])

            # Filter data berdasarkan tanggal
            in_sample_data = data[(data['Date'] >= in_sample_start) & (data['Date'] <= in_sample_end)]
            out_sample_data = data[(data['Date'] >= out_sample_start) & (data['Date'] <= out_sample_end)]

            # Simpan hasil split ke CSV
            in_sample_data.to_csv('in_sample_data.csv', index=False)
            out_sample_data.to_csv('out_sample_data.csv', index=False)

            # Ambil 5 baris pertama dan terakhir untuk ditampilkan di HTML
            in_sample_head = in_sample_data.head(5).to_dict(orient='records')
            in_sample_tail = in_sample_data.tail(5).to_dict(orient='records')
            out_sample_head = out_sample_data.head(5).to_dict(orient='records')
            out_sample_tail = out_sample_data.tail(5).to_dict(orient='records')

            # Simpan hasil ke session
            session['in_sample_head'] = in_sample_head
            session['out_sample_head'] = out_sample_head
            session['in_sample_tail'] = in_sample_tail
            session['out_sample_tail'] = out_sample_tail
            session['in_sample_shape'] = in_sample_data.shape
            session['out_sample_shape'] = out_sample_data.shape

            return render_template(
                'split_data.html',
                in_sample_shape=in_sample_data.shape,
                out_sample_shape=out_sample_data.shape,
                in_sample_head=in_sample_head,
                in_sample_tail=in_sample_tail,
                out_sample_head=out_sample_head,
                out_sample_tail=out_sample_tail,
                success="Data telah berhasil di-split dan disimpan!",
                current_page='split_data'
            )

        except Exception as e:
            return render_template('split_data.html', error=f"Terjadi kesalahan: {e}", current_page='split_data')

    return render_template('split_data.html', in_sample_head=in_sample_head, out_sample_head=out_sample_head, in_sample_tail=in_sample_tail, out_sample_tail=out_sample_tail, in_sample_shape=in_sample_shape, out_sample_shape=out_sample_shape, current_page='split_data')

@app.route('/model_ar', methods=['GET', 'POST'])
def identifikasi_ar():
    # Pastikan file in_sample_data.csv tersedia
    if not os.path.exists('in_sample_data.csv'):
        return render_template('model_ar.html', error="File in_sample_data.csv tidak ditemukan.", current_page='model_ar')

    # Load data in-sample
    data = pd.read_csv('in_sample_data.csv')
    data['Date'] = pd.to_datetime(data['Date'])

    if 'Return' not in data.columns:
        return render_template('model_ar.html', error="Kolom 'Return' tidak ditemukan dalam data.", current_page='model_ar')

    try:
        returns = data['Return']
        lags = 30  # Jumlah lag yang ditampilkan

        # 🔹 Hitung nilai ACF & PACF
        acf_values = acf(returns, nlags=lags)
        pacf_values = pacf(returns, nlags=lags, method='ywm')

        # 🔹 Simpan gambar ACF & PACF
        fig, ax = plt.subplots(2, 1, figsize=(10, 8))

        ax[0].bar(range(lags + 1), acf_values, width=0.5, color='blue', alpha=0.7)
        ax[0].axhline(y=0, linestyle='--', color='black', linewidth=1)
        ax[0].set_title("Autocorrelation (Histogram)", fontsize=12)

        ax[1].bar(range(lags + 1), pacf_values, width=0.5, color='red', alpha=0.7)
        ax[1].axhline(y=0, linestyle='--', color='black', linewidth=1)
        ax[1].set_title("Partial Autocorrelation (Histogram)", fontsize=12)

        plt.suptitle("Data Saham ACF & PACF", fontsize=12, x=0.515, y=0.98)
        plt.tight_layout()

        # 🔹 Simpan ke dalam folder static/uploads/
        img_path = "static/uploads/acf_pacf_plot.png"
        os.makedirs(os.path.dirname(img_path), exist_ok=True)
        plt.savefig(img_path)
        plt.close()  # Tutup plot agar tidak ditampilkan di layar

        # 🔹 Simpan hasil dalam bentuk dictionary
        acf_dict = {f"Lag {i}": round(acf_values[i], 4) for i in range(lags + 1)}
        pacf_dict = {f"Lag {i}": round(pacf_values[i], 4) for i in range(lags + 1)}

        # 🔹 Identifikasi model AR terbaik berdasarkan AIC
        warnings.filterwarnings("ignore")
        max_p = 10
        aic_values = []

        for p in range(1, max_p + 1):
            try:
                model = ARIMA(returns, order=(p, 0, 0))
                model_fit = model.fit()
                aic_values.append((p, model_fit.aic))
            except:
                continue

        best_p, best_aic = min(aic_values, key=lambda x: x[1])

        return render_template(
            'model_ar.html',
            acf_dict=acf_dict,
            pacf_dict=pacf_dict,
            best_p=best_p,
            best_aic=best_aic,
            acf_pacf_image=img_path,
            current_page='model_ar'
        )

    except Exception as e:
        return render_template('model_ar.html', error=f"Terjadi kesalahan: {e}", current_page='model_ar')

@app.route('/asumsi_ar', methods=['GET', 'POST'])
def asumsi_klasik_ar():
    error = None
    ks_statistic = ks_pvalue = W_critical = None
    dw_stat = None
    lm_stat = lm_pvalue = t_critical_arch = None
    dw_lower, dw_upper = 1.5, 2.5  # Rentang kritis DW
    arima_summary = None

    # Pastikan file in_sample_data.csv tersedia
    if not os.path.exists('in_sample_data.csv'):
        return render_template('asumsi_ar.html', error="File in_sample_data.csv tidak ditemukan.", current_page='asumsi_ar')

    # Load data
    data = pd.read_csv('in_sample_data.csv')
    
    if 'Return' not in data.columns:
        return render_template('asumsi_ar.html', error="Kolom 'Return' tidak ditemukan dalam data.", current_page='asumsi_ar')
    
    # Ambil input p dari user
    p = request.form.get("p")
    
    if p:
        try:
            p = int(p)
            if p < 1 or p > 10:  # Batasan nilai p
                raise ValueError("Nilai p harus antara 1 dan 10.")

            # Estimasi model ARIMA dengan p dan d=q=0
            model_arima = ARIMA(data['Return'], order=(p, 0, 0)).fit()
            arima_summary = model_arima.summary()

            residuals = model_arima.resid
            n = len(residuals)

            # Uji Normalitas (Kolmogorov-Smirnov)
            ks_statistic, ks_pvalue = kstest(residuals, 'norm', args=(np.mean(residuals), np.std(residuals)))
            W_critical = 1.36 / np.sqrt(n)

            # Uji Autokorelasi (Durbin-Watson)
            dw_stat = durbin_watson(residuals)

            # Uji Homoskedastisitas (ARCH LM Test)
            def arch_lm_test(residuals, m):
                n = len(residuals)
                residuals_sq = residuals ** 2
                SSR1 = np.sum(residuals_sq)
                omega_bar = np.mean(residuals_sq)
                SSR0 = np.sum((residuals_sq - omega_bar) ** 2)
                LM_stat = ((SSR0 - SSR1) / m) / (SSR1 / (n - 2*m - 1))
                p_value = 1 - chi2.cdf(LM_stat, m)
                return LM_stat, p_value

            m = 1  # Jumlah lag untuk ARCH LM Test
            lm_stat, lm_pvalue = arch_lm_test(residuals, m)
            t_critical_arch = chi2.ppf(1 - 0.05, df=m)

        except ValueError as e:
            error = f"Input tidak valid: {e}"

    return render_template(
        'asumsi_ar.html',
        error=error,
        p=p,
        ks_statistic=ks_statistic,
        ks_pvalue=ks_pvalue,
        W_critical=W_critical,
        dw_stat=dw_stat,
        dw_lower=dw_lower,
        dw_upper=dw_upper,
        lm_stat=lm_stat,
        lm_pvalue=lm_pvalue,
        t_critical_arch=t_critical_arch, 
        arima_summary=arima_summary,
        current_page='asumsi_ar'
    )

@app.route('/linearitas', methods=['GET', 'POST'])
def uji_linearitas():
    error = None
    LM3 = chi_square_critical = p_value = None
    kesimpulan = ""
    summary_aux = None
    T = SSR_0 = SSR_1 = None
    table_preview = None

    # Pastikan file data tersedia
    if not os.path.exists('in_sample_data.csv'):
        return render_template('linearitas.html', error="File in_sample_data.csv tidak ditemukan.",  current_page='linearitas')

    # Load data
    data = pd.read_csv('in_sample_data.csv')

    if 'Return' not in data.columns:
        return render_template('linearitas.html', error="Kolom 'Return' tidak ditemukan dalam data.",  current_page='linearitas')

    # Buat tabel preview 5 baris pertama
    # table_preview = data.head().to_html(classes="table table-bordered", index=False)

    # Ambil input p dari user
    p = request.form.get("p")

    if p:
        try:
            p = int(p)
            if p < 1 or p > 10:  # Batasi nilai p
                raise ValueError("Nilai p harus antara 1 dan 10.")

            # **Step 1: Estimasi Model AR**
            model = sm.tsa.ARIMA(data['Return'], order=(p, 0, 0))
            model_fit = model.fit()
            residuals = model_fit.resid

            # **Step 2: Regresi Bantu**
            data_lm = pd.DataFrame({
                'Return': data['Return'],
                'Lagged_Return': data['Return'].shift(1)
            }).dropna()

            X_ar = sm.add_constant(data_lm['Lagged_Return'])
            y_ar = data_lm['Return']
            ar_model = sm.OLS(y_ar, X_ar).fit()
            residuals = ar_model.resid

            SSR_0 = np.sum(residuals ** 2)

            # **Step 3: Buat variabel tambahan**
            data_lm['Residuals_Squared'] = residuals ** 2
            data_lm['Lagged_Return_Squared'] = data_lm['Lagged_Return'] ** 2
            data_lm['Lagged_Return_Cubed'] = data_lm['Lagged_Return'] ** 3

            X_aux = sm.add_constant(data_lm[['Lagged_Return', 'Lagged_Return_Squared', 'Lagged_Return_Cubed']])
            y_aux = data_lm['Residuals_Squared']
            aux_model = sm.OLS(y_aux, X_aux).fit()
            residuals_aux = aux_model.resid

            # Buat tabel preview 5 baris pertama
            table_preview = data_lm.head().to_html(classes="table table-bordered", index=False)

            SSR_1 = np.sum(residuals_aux ** 2)

            # **Step 4: Hitung LM3**
            T = len(data_lm)
            LM3 = T * (SSR_0 - SSR_1) / SSR_0

            # **Step 5: Uji dengan Distribusi Chi-Square**
            df = 3 * (p + 1)
            alpha = 0.95
            chi_square_critical = chi2.ppf(alpha, df)
            p_value = chi2.sf(LM3, df)

            # **Step 6: Kesimpulan**
            if LM3 > chi_square_critical:
                kesimpulan = "Tolak H0: Ada indikasi non-linearitas, model non-linear diperlukan."
            else:
                kesimpulan = "Terima H0: Tidak ada indikasi non-linearitas, model linear cukup."

            # Simpan hasil OLS summary dalam HTML format
            summary_aux = aux_model.summary().as_html()

        except ValueError as e:
            error = f"Input tidak valid: {e}"

    return render_template(
        'linearitas.html',
        error=error,
        p=p,
        T=T,
        SSR_0=SSR_0,
        SSR_1=SSR_1,
        LM3=LM3,
        chi_square_critical=chi_square_critical,
        p_value=p_value,
        kesimpulan=kesimpulan,
        summary_aux=summary_aux,
        table_preview=table_preview,
        current_page='linearitas'
    )

@app.route('/fungsi_transisi', methods=['GET', 'POST'])
def fungsi_transisi():
    error = None
    summary_aux = None
    model_selected = ""
    table_preview = None
    keputusan_t = {}
    keputusan_p = {}

    if not os.path.exists('in_sample_data.csv'):
        return render_template('fungsi_transisi.html', error="File in_sample_data.csv tidak ditemukan.", current_page='fungsi_transisi')

    data_model = pd.read_csv('in_sample_data.csv')

    if 'Return' not in data_model.columns:
        return render_template('fungsi_transisi.html', error="Kolom 'Return' tidak ditemukan dalam data.", current_page='fungsi_transisi')

    try:
        data_model['Lagged_Return'] = data_model['Return'].shift(1)
        data_model = data_model.dropna()

        X_ar = sm.add_constant(data_model['Lagged_Return'])
        y_ar = data_model['Return']
        ar_model = sm.OLS(y_ar, X_ar).fit()
        residuals = ar_model.resid

        data_model['Residuals_Squared'] = residuals ** 2
        data_model['Lagged_Return_Squared'] = data_model['Lagged_Return'] ** 2
        data_model['Lagged_Return_Cubed'] = data_model['Lagged_Return'] ** 3

        X_aux = sm.add_constant(data_model[['Lagged_Return', 'Lagged_Return_Squared', 'Lagged_Return_Cubed']])
        y_aux = data_model['Residuals_Squared']
        aux_model = sm.OLS(y_aux, X_aux).fit()

        summary_aux = aux_model.summary().as_html()

        t_stat_3 = aux_model.tvalues['Lagged_Return_Cubed']
        p_value_3 = aux_model.pvalues['Lagged_Return_Cubed']
        t_stat_2 = aux_model.tvalues['Lagged_Return_Squared']
        p_value_2 = aux_model.pvalues['Lagged_Return_Squared']
        t_stat_1 = aux_model.tvalues['Lagged_Return']
        p_value_1 = aux_model.pvalues['Lagged_Return']

        df = len(data_model) - int(aux_model.df_model)
        alpha = 0.05
        t_critical = t.ppf(1 - alpha / 2, df)

        # Keputusan berdasarkan t-Statistic
        keputusan_t["H03"] = "Tolak H03" if abs(t_stat_3) > t_critical else "Gagal Tolak H03"
        keputusan_t["H02"] = "Tolak H02" if abs(t_stat_2) > t_critical else "Gagal Tolak H02"
        keputusan_t["H01"] = "Tolak H01" if abs(t_stat_1) > t_critical else "Gagal Tolak H01"

        # Keputusan berdasarkan p-Value
        keputusan_p["H03"] = "Tolak H03" if p_value_3 < alpha else "Gagal Tolak H03"
        keputusan_p["H02"] = "Tolak H02" if p_value_2 < alpha else "Gagal Tolak H02"
        keputusan_p["H01"] = "Tolak H01" if p_value_1 < alpha else "Gagal Tolak H01"

        # Pemilihan model berdasarkan uji hipotesis
        if p_value_3 < alpha:
            model_selected = "Pilih model LSTAR (β3 ≠ 0)."
        elif p_value_3 > alpha and p_value_2 < alpha:
            model_selected = "Pilih model ESTAR (β3 = 0, β2 ≠ 0)."
        elif p_value_3 > alpha and p_value_2 > alpha and p_value_1 < alpha:
            model_selected = "Pilih model LSTAR (β1 ≠ 0)."
        else:
            model_selected = "Pilih model ESTAR (β3 = 0, β2 = 0, β1 = 0)."

    except Exception as e:
        error = f"Terjadi kesalahan: {str(e)}"

    return render_template(
        'fungsi_transisi.html',
        error=error,
        summary_aux=summary_aux,
        t_critical= t_critical,
        t_stat_1=t_stat_1, p_value_1=p_value_1,
        t_stat_2=t_stat_2, p_value_2=p_value_2,
        t_stat_3=t_stat_3, p_value_3=p_value_3,
        keputusan_t=keputusan_t,
        keputusan_p=keputusan_p,
        model_selected=model_selected,
        current_page='fungsi_transisi'
    )

@app.route('/estimasi_dan_asumsi_estar', methods=['GET', 'POST'])
def estimasi_dan_asumsi_estar():
    if request.method == 'POST':
        try:
            # Ambil nilai p dan d dari form
            p = int(request.form['p'])
            d = int(request.form['d'])

            if p <= 0 or d <= 0:
                return jsonify({'error': "Nilai p dan d harus lebih besar dari 0."})

            # Load dataset
            data_model = pd.read_csv('in_sample_data.csv')
            if 'Return' not in data_model.columns:
                return jsonify({'error': "Kolom 'Return' tidak ditemukan dalam data."})

            # Fungsi untuk menyiapkan data
            def prepare_data(data, p, d):
                """Membuat lag berdasarkan ordo (p,d)"""
                for i in range(1, p + 1):
                    data[f'Lag{i}'] = data['Return'].shift(i)
                data['LagT'] = data['Return'].shift(d)
                return data.dropna()

            # Fungsi transisi ESTAR
            def ESTAR_transition(gamma, threshold, yt_lagged):
                """Fungsi transisi eksponensial smooth transition"""
                return 1 - np.exp(-gamma * (yt_lagged - threshold) ** 2)

            # Model ESTAR
            def ESTAR_model(params, y, lags, yt_lagT):
                """Model ESTAR dengan parameter fleksibel berdasarkan p"""
                const_L = params[0]
                phiL = params[1:p + 1]  # Koefisien AR pada rezim rendah
                const_H = params[p + 1]
                phiH = params[p + 2:2 * p + 2]  # Koefisien AR pada rezim tinggi
                gamma = params[2 * p + 2]
                threshold = params[2 * p + 3]

                transition = ESTAR_transition(gamma, threshold, yt_lagT)

                yt_L = const_L + np.dot(lags, phiL)  # Rezim rendah
                yt_H = const_H + np.dot(lags, phiH)  # Rezim tinggi

                return (1 - transition) * yt_L + transition * yt_H

            # Fungsi loss untuk optimasi
            def nls_loss(params, y, lags, yt_lagT):
                """Fungsi loss untuk optimasi NLS"""
                y_hat = ESTAR_model(params, y, lags, yt_lagT)
                return np.sum((y - y_hat) ** 2)

            # Estimasi model ESTAR
            def estimate_ESTAR(data, p, d):
                """Estimasi model ESTAR dengan input (p,d)"""
                data = prepare_data(data, p, d)

                lags = data[[f'Lag{i}' for i in range(1, p + 1)]]
                yt_lagT = data['LagT']
                y = data['Return']

                params_initial = np.random.uniform(-0.5, 0.5, size=2 * p + 4)  # Inisialisasi acak
                result = minimize(nls_loss, params_initial, args=(y, lags, yt_lagT), method='BFGS')

                params_estimated = result.x
                hessian_inv = result.hess_inv
                standard_errors = np.sqrt(np.diag(hessian_inv))
                t_values = params_estimated / standard_errors
                p_values = 2 * (1 - t_dist.cdf(np.abs(t_values), df=len(data) - len(params_estimated)))

                param_names = ['Const_L'] + [f'phiL_{i}' for i in range(1, p + 1)] + \
                              ['Const_H'] + [f'phiH_{i}' for i in range(1, p + 1)] + ['gamma', 'threshold']

                results_df = pd.DataFrame({
                    'Parameter': param_names,
                    'Estimate': params_estimated,
                    'Standard Error': standard_errors,
                    't-value': t_values,
                    'p-value': p_values
                })

                # Simpan hasil estimasi ke CSV
                results_df.to_csv('params_estimated.csv', index=False)

                return results_df, params_estimated

            # Estimasi Model
            results_df, params_estimated = estimate_ESTAR(data_model, p, d)

            # Menyiapkan data untuk perhitungan residuals
            data_model = prepare_data(data_model, p, d)  # Pastikan data sudah dipersiapkan
            y = data_model['Return'].iloc[d:]  # Mengambil y yang sesuai
            lags = data_model[[f'Lag{i}' for i in range(1, p + 1)]].iloc[d:]  # Menggunakan iloc
            yt_lagT = data_model['LagT'].iloc[d:]  # Menggunakan iloc

            y_hat = ESTAR_model(params_estimated, y, lags, yt_lagT)
            residuals = y - y_hat  # Pastikan y dan y_hat memiliki panjang yang sama
            
            # Uji Durbin-Watson
            dw_stat = durbin_watson(residuals)
            dw_lower, dw_upper = 1.5, 2.5

            # Uji ARCH-LM
            def arch_lm_test(residuals, m):
                n = len(residuals)
                residuals_sq = residuals ** 2
                SSR1 = np.sum(residuals_sq)
                omega_bar = np.mean(residuals_sq)
                SSR0 = np.sum((residuals_sq - omega_bar) ** 2)
                LM_stat = ((SSR0 - SSR1) / m) / (SSR1 / (n - 2 * m - 1))
                p_value = 1 - chi2.cdf(LM_stat, m)
                return LM_stat, p_value

            m = 1  # Jumlah lag untuk uji ARCH-LM
            lm_stat, lm_pval = arch_lm_test(residuals, m)
            t_critical_arch = chi2.ppf(1 - 0.05, df=m)

            # Uji Kolmogorov-Smirnov
            ks_statistic, ks_pvalue = kstest(residuals, 'norm', args=(np.mean(residuals), np.std(residuals)))
            n = len(residuals)
            ks_critical = 1.36 / np.sqrt(n)  # D-Critical dari tabel Kolmogorov-Smirnov

            return render_template('estimasi_dan_asumsi_estar.html',
                                   results=results_df.to_dict(orient='records'),
                                   dw_stat=dw_stat, dw_result=(dw_lower < dw_stat < dw_upper),
                                   lm_stat=lm_stat, lm_pval=lm_pval, t_critical_arch=t_critical_arch,
                                   arch_result=(lm_stat <= t_critical_arch and lm_pval > 0.05),
                                   ks_statistic=ks_statistic, ks_pvalue=ks_pvalue, ks_critical=ks_critical,
                                   ks_result=(ks_statistic < ks_critical and ks_pvalue > 0.05),
                                   current_page='estimasi_dan_asumsi_estar')

        except Exception as e:
            return jsonify({'error': str(e)})

    return render_template('estimasi_dan_asumsi_estar.html', results=None, current_page='estimasi_dan_asumsi_estar')

predicted_prices = []

@app.route('/prediksi_estar', methods=['GET', 'POST'])
def prediksi_estar():
    global predicted_prices

    predicted_returns = None  # Inisialisasi

    if request.method == 'POST':
        p = int(request.form['p'])
        d = int(request.form['d'])
        steps = int(request.form['steps'])
        last_price = float(request.form['last_price'])

        params_df = pd.read_csv('params_estimated.csv')
        params_estimated = params_df['Estimate'].values

        data_lag = pd.DataFrame({f'Lag{i}': [0] for i in range(1, p + 1)})

        def ESTAR_forecast(data, params_array, p, steps=5):
            params = {
                'Const_L': params_array[0],
                'phiL': params_array[1:p + 1],
                'Const_H': params_array[p + 1],
                'phiH': params_array[p + 2:2 * p + 2],
                'gamma': params_array[2 * p + 2],
                'threshold': params_array[2 * p + 3]
            }
            forecasts = []
            for step in range(steps):
                lag_vals = data.iloc[-1][[f'Lag{i}' for i in range(1, p + 1)]].values
                th = params['threshold']
                gamma = params['gamma']
                G = 1 - np.exp(-gamma * (lag_vals[-1] - th) ** 2)

                f_L = params['Const_L'] + np.dot(params['phiL'], lag_vals)
                f_H = params['Const_H'] + np.dot(params['phiH'], lag_vals)

                forecast = (1 - G) * f_L + G * f_H
                forecasts.append(forecast)

                new_row = {f'Lag{i}': forecast for i in range(1, p + 1)}
                data = pd.concat([data, pd.DataFrame([new_row])], ignore_index=True)

            return forecasts

        # Menjalankan peramalan menggunakan ESTAR
        forecasts = ESTAR_forecast(data_lag, params_estimated, p, steps)

        # Menghitung harga prediksi berdasarkan return forecast
        forecast_prices = [last_price]
        for forecast in forecasts:
            next_price = forecast_prices[-1] * (1 + forecast)
            forecast_prices.append(next_price)

        # Save results to CSV
        forecast_df = pd.DataFrame({
            'Step': range(1, len(forecast_prices)),
            'Forecast_Return': forecasts,
            'Forecast_Close_Price': forecast_prices[1:],
        })
        forecast_df.to_csv('forecast_results.csv', index=False)

        predicted_prices = forecast_prices[1:]

        # Menyimpan hasil untuk ditampilkan
        predicted_returns = forecasts  # Simpan hasil prediksi return

    return render_template('prediksi_estar.html', predicted_returns=predicted_returns, predicted_prices=predicted_prices, current_page='prediksi_estar')

@app.route('/mape_model', methods=['GET'])
def mape_model():
    # Load processed return data
    data_all = pd.read_csv(os.path.join(UPLOAD_FOLDER, 'return_processed_data.csv'))
    close_price = data_all['Price'].values

    # Load data in-sample
    dataload = pd.read_csv('in_sample_data.csv')

    # Load data out-of-sample
    data_out = pd.read_csv('out_sample_data.csv')
    data_out = data_out.reset_index(drop=True)

    # Load forecast results from CSV
    forecast_df = pd.read_csv('forecast_results.csv')
    forecast_df['Actual_Price'] = data_out['Price'].iloc[:len(forecast_df)].values
    forecast_df['Date'] = data_out['Date'].iloc[:len(forecast_df)].values

    # Extract actual prices for the second plot
    actual_prices_estimation = dataload['Price'].values

    # Calculate MAPE
    def calculate_mape(actual, forecast):
        actual = np.array(actual)
        forecast = np.array(forecast)
        mape = np.mean(np.abs((actual - forecast) / actual)) * 100
        return mape

    predicted_prices = forecast_df['Forecast_Close_Price']
    mape = calculate_mape(forecast_df['Actual_Price'], predicted_prices)

    # Simpan MAPE ke dalam DataFrame
    # Pastikan MAPE adalah float
    mape = float(mape)
    mape_df = pd.DataFrame({'mape': [mape]})
    mape_df.to_csv(os.path.join(UPLOAD_FOLDER, 'mape_result.csv'), index=False)

    # Plot harga aktual vs prediksi
    plt.figure(figsize=(10, 6))
    plt.plot(forecast_df['Actual_Price'], label='Harga Aktual', marker='o')
    plt.plot(predicted_prices, label='Harga Prediksi', marker='o')
    plt.title("Perbandingan Harga Aktual vs Prediksi (ESTAR)")
    plt.xlabel("Waktu (Step)")
    plt.ylabel("Harga Close")
    plt.legend()
    plt.grid(True)
    plt.savefig('static/actual_vs_predicted.png')
    plt.close()

    # Simulasi data aktual dan prediksi untuk grafik kedua
    plt.figure(figsize=(12, 6))
    plt.plot(range(len(actual_prices_estimation)), actual_prices_estimation, label="Harga Aktual", color="blue", linewidth=2)
    plt.plot(range(len(actual_prices_estimation) - 1, len(actual_prices_estimation) - 1 + len(predicted_prices)),
             predicted_prices, label="Harga Prediksi (ESTAR)", color="red", linestyle="--", linewidth=2)
    plt.title("Prediksi Harga Saham Menggunakan Model ESTAR", fontsize=16)
    plt.xlabel("Waktu (Langkah ke-)", fontsize=14)
    plt.ylabel("Harga Saham", fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(alpha=0.5)
    plt.tight_layout()
    plt.savefig('static/forecast_prices.png')
    plt.close()

    # Simulasi data aktual dan prediksi untuk grafik kedua
    plt.figure(figsize=(12, 6))
    plt.plot(range(len(close_price)), close_price, label="Harga Aktual", color="blue", linewidth=2)
    plt.title("Harga Saham Keseluruhan Data", fontsize=16)
    plt.xlabel("Waktu (Langkah ke-)", fontsize=14)
    plt.ylabel("Harga Saham", fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(alpha=0.5)
    plt.tight_layout()
    plt.savefig('static/close_price_all.png')
    plt.close()

    # Select and display specific columns
    result_df = forecast_df[['Date', 'Actual_Price', 'Forecast_Close_Price']]

    return render_template('mape_model.html', mape=mape, result_df=result_df, current_page='mape_model')

@app.route('/prediksi_estar_utk_var', methods=['GET', 'POST'])
def prediksi_estar_utk_var():
    global predicted_prices

    predicted_returns = None  # Inisialisasi
    predicted_prices = None
    remaining_returns = None
    remaining_prices = None

    if request.method == 'POST':
        p = int(request.form['p'])
        d = int(request.form['d'])
        steps = int(request.form['steps'])
        outsample_count = int(request.form['outsample_count'])  # Ambil input jumlah out-of-sample
        last_price = float(request.form['last_price'])

        steps = steps + outsample_count

        params_df = pd.read_csv('params_estimated.csv')
        params_estimated = params_df['Estimate'].values

        data_lag = pd.DataFrame({f'Lag{i}': [0] for i in range(1, p + 1)})

        def ESTAR_forecast(data, params_array, p, steps):
            params = {
                'Const_L': params_array[0],
                'phiL': params_array[1:p + 1],
                'Const_H': params_array[p + 1],
                'phiH': params_array[p + 2:2 * p + 2],
                'gamma': params_array[2 * p + 2],
                'threshold': params_array[2 * p + 3]
            }
            forecasts = []
            for step in range(steps):
                lag_vals = data.iloc[-1][[f'Lag{i}' for i in range(1, p + 1)]].values
                th = params['threshold']
                gamma = params['gamma']
                G = 1 - np.exp(-gamma * (lag_vals[-1] - th) ** 2)

                f_L = params['Const_L'] + np.dot(params['phiL'], lag_vals)
                f_H = params['Const_H'] + np.dot(params['phiH'], lag_vals)

                forecast = (1 - G) * f_L + G * f_H
                forecasts.append(forecast)

                new_row = {f'Lag{i}': forecast for i in range(1, p + 1)}
                data = pd.concat([data, pd.DataFrame([new_row])], ignore_index=True)

            return forecasts

        # Menjalankan peramalan menggunakan ESTAR
        forecasts = ESTAR_forecast(data_lag, params_estimated, p, steps)

        # Menghitung harga prediksi berdasarkan return forecast
        forecast_prices = [last_price]
        for forecast in forecasts:
            next_price = forecast_prices[-1] * (1 + forecast)
            forecast_prices.append(next_price)

        # Save results to CSV
        forecast_df = pd.DataFrame({
            'Step': range(1, len(forecast_prices)),
            'Forecast_Return': forecasts,
            'Forecast_Close_Price': forecast_prices[1:],
        })
        forecast_df.to_csv('forecast_results_new.csv', index=False)

        predicted_prices = forecast_prices[1:]
        predicted_returns = forecasts  # Simpan hasil prediksi return

        # Simpan sisa data setelah mengurangi jumlah data out-of-sample
        remaining_data = len(predicted_prices) - outsample_count
        if remaining_data > 0:
            remaining_returns = predicted_returns[outsample_count:]
            remaining_prices = predicted_prices[outsample_count:]

            remaining_df = pd.DataFrame({
                'Step': range(1, len(remaining_returns) + 1),
                'Forecast_Return': remaining_returns,
                'Forecast_Close_Price': remaining_prices
            })
            remaining_df.to_csv('remaining_forecast_results.csv', index=False)

            # Simulasi visualisasi perbandingan harga aktual dan prediksi
            data_out = pd.read_csv('out_sample_data.csv')
            data_out = data_out.reset_index(drop=True)

            # Load data in-sample
            data_in = pd.read_csv('in_sample_data.csv')
            # Extract actual prices for the second plot
            actual_prices_estimation = data_in['Price'].values

            # Plot harga aktual vs prediksi
            plt.figure(figsize=(10, 6))
            plt.plot(data_out['Price'].iloc[:len(predicted_prices)], label='Harga Aktual', marker='o')
            plt.plot(predicted_prices, label='Harga Prediksi', marker='o')
            plt.title("Perbandingan Harga Aktual vs Prediksi (ESTAR)")
            plt.xlabel("Waktu (Step)")
            plt.ylabel("Harga Close")
            plt.legend()
            plt.grid(True)
            plt.savefig('static/actual_vs_predicted_var.png')
            plt.close()

            # Simulasi data aktual dan prediksi untuk grafik kedua
            plt.figure(figsize=(12, 6))
            plt.plot(range(len(actual_prices_estimation)), actual_prices_estimation, label="Harga Aktual", color="blue", linewidth=2)
            plt.plot(range(len(actual_prices_estimation) - 1, len(actual_prices_estimation) - 1 + len(predicted_prices)),
                     predicted_prices, label="Harga Prediksi (ESTAR)", color="red", linestyle="--", linewidth=2)
            plt.title("Prediksi Harga Saham Menggunakan Model ESTAR", fontsize=16)
            plt.xlabel("Waktu (Langkah ke-)", fontsize=14)
            plt.ylabel("Harga Saham", fontsize=14)
            plt.legend(fontsize=12)
            plt.grid(alpha=0.5)
            plt.tight_layout()
            plt.savefig('static/forecast_prices_var.png')
            plt.close()

    return render_template('prediksi_estar_utk_var.html', 
                           remaining_returns=remaining_returns, 
                           remaining_prices=remaining_prices, 
                           current_page='prediksi_estar_utk_var')

@app.route('/var', methods=['GET', 'POST'])
def var():
    var_result = None
    p_alpha = None

    if request.method == 'POST':
        initial_investment = float(request.form['initial_investment'])
        holding_period = int(request.form['holding_period'])
        confidence_level = float(request.form['confidence_level']) / 100  # Ambil dari input dan ubah ke decimal
        source = request.form['source']  # Mendapatkan sumber prediksi

        # Memilih file data berdasarkan sumber
        if source == 'prediksi_estar':
            data_var = pd.read_csv('forecast_results_new.csv')
        else:
            data_var = pd.read_csv('remaining_forecast_results.csv')

        # Ambil data return dari kolom yang sesuai
        data_return = data_var['Forecast_Return']

        # Fungsi untuk menghitung VaR dengan metode simulasi historis
        def historical_var(returns, confidence_level):
            P_alpha = np.percentile(returns, (1 - confidence_level) * 100)
            return P_alpha

        # Menghitung persentil return (P_alpha)
        p_alpha = historical_var(data_return, confidence_level=confidence_level)

        # Menghitung VaR berdasarkan rumus yang digunakan
        var_holding = initial_investment * abs(p_alpha) * np.sqrt(holding_period)

        # Simpan hasil dalam bentuk dictionary
        var_result = {
            'p_alpha': p_alpha,
            'var_holding': var_holding,
            'holding_period': holding_period,
            'confidence_level': confidence_level * 100
        }

        # Konversi var_result ke DataFrame dan simpan ke CSV
        var_result_df = pd.DataFrame([var_result])
        var_result_df.to_csv(os.path.join(UPLOAD_FOLDER, 'var_result.csv'), index=False)
        
    return render_template('var.html', var_result=var_result, current_page='var')

@app.route('/interpretasi', methods=['GET', 'POST'])
def interpretasi():
    df_mape = pd.read_csv(os.path.join(UPLOAD_FOLDER, 'mape_result.csv'))
    df_var = pd.read_csv(os.path.join(UPLOAD_FOLDER, 'var_result.csv'))

    # Inisialisasi variabel hasil
    mape_value = None
    holding_period = None
    var_holding = None
    p_alpha = None
    confidence_level = None
    forecast_data_estar = []
    forecast_data_new = []

    # Memuat hasil MAPE
    try:
        mape_value = df_mape['mape'].iloc[0] if not df_mape.empty else None
    except Exception as e:
        print(f"Error saat membaca mape_result.csv: {e}")

    # Memuat hasil VaR
    try:
        holding_period = df_var['holding_period'].iloc[0] if not df_var.empty else None
        var_holding = df_var['var_holding'].iloc[0] if not df_var.empty else None
        p_alpha = df_var['p_alpha'].iloc[0] if not df_var.empty else None
        confidence_level = df_var['confidence_level'].iloc[0] if not df_var.empty else None
    except Exception as e:
        print(f"Error saat membaca var_result.csv: {e}")

    # Memuat data prediksi
    try:
        df_forecast_estar = pd.read_csv('forecast_results.csv')
        forecast_data_estar = df_forecast_estar.to_dict(orient='records')
    except Exception as e:
        print(f"Error saat membaca forecast_results.csv: {e}")

    try:
        df_forecast_new = pd.read_csv('forecast_results_new.csv')
        forecast_data_new = df_forecast_new.to_dict(orient='records')
    except Exception as e:
        print(f"Error saat membaca forecast_results_new.csv: {e}")

    # Menentukan saran berdasarkan hasil prediksi
    if forecast_data_estar and forecast_data_new:
        predicted_return_estar = forecast_data_estar[-1]['Forecast_Return']
        predicted_return_new = forecast_data_new[-1]['Forecast_Return']

        if predicted_return_new > 0.1:  # Misalnya, jika return > 10%
            recommendation = "Strong Buy"
        elif predicted_return_new > 0:  # Jika return > 0%
            recommendation = "Buy"
        elif predicted_return_new == 0:  # Jika return = 0%
            recommendation = "Hold"
        elif predicted_return_new < 0 and predicted_return_new > -0.1:  # Jika return antara -10% hingga 0%
            recommendation = "Sell"
        else:  # Jika return < -10%
            recommendation = "Strong Sell"

    return render_template('interpretasi.html', 
                           forecast_data_estar=forecast_data_estar, 
                           forecast_data_new=forecast_data_new,
                           mape_value=mape_value,
                           holding_period=holding_period, 
                           var_holding=var_holding, 
                           p_alpha=p_alpha, 
                           confidence_level=confidence_level, 
                           recommendation=recommendation,
                           current_page='interpretasi')

# Main Program
if __name__ == '__main__':
    app.run(debug=True)