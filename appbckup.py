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

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        if not file:
            return render_template('upload.html', error="Tidak ada file yang diunggah.")

        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(file_path)

        try:
            data = pd.read_csv(file_path)
            data = data.head(15).to_dict(orient='records')
            return render_template('upload.html', data=data, file_name=file.filename)
        except Exception as e:
            return render_template('upload.html', error=f"Terjadi kesalahan saat membaca file: {e}")

    return render_template('upload.html')

@app.route('/prepro', methods=['GET', 'POST'])
def preprocess_data():
    if request.method == 'POST':
        # Pastikan file_name ada dalam request.form
        file_name = request.form.get('file_name')
        
        if not file_name:
            return render_template('prepro.html', error="Nama file diperlukan.", img_harga=None)

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

            return render_template('prepro.html',
                                   null_counts=null_counts,
                                   duplicate_counts=duplicate_counts,
                                   data_description=data_description,
                                   processed_data=dataload.head(15).to_dict(orient='records'),
                                   img_harga=plot_url)  # Pass the plot URL to template

        except Exception as e:
            return render_template('prepro.html', error=f"Terjadi kesalahan saat preprocessing data: {e}", img_harga=plot_url)
    
    # Jika metode GET, hanya render template tanpa melakukan pemrosesan
    return render_template('prepro.html', img_harga=None)

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

    return render_template('return.html', data=dataload_top15.to_dict(orient='records'), img_return=img_return)

def create_return_plot(dataload):
    plt.figure(figsize=(16, 5))
    plt.plot(dataload.index, dataload['Return'], label="Return", color='blue')
    plt.legend(loc='best')
    plt.title("Return Data BBCA")
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
                return render_template('stasioneritas.html', error="Data tidak ditemukan. Silakan lakukan preprocessing terlebih dahulu.")

        # 🔹 Pastikan kolom 'Return' ada
        if 'Return' not in dataload.columns:
            return render_template('stasioneritas.html', error="Kolom 'Return' belum dihitung. Silakan hitung return terlebih dahulu.")

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

        return render_template('stasioneritas.html', adf_output=adf_results)

    return render_template('stasioneritas.html')

@app.route('/split_data', methods=['GET', 'POST'])
def split_data():
    if not os.path.exists(RETURN_FILE):
        return render_template('split_data.html', error="File return_processed_data.csv tidak ditemukan.")

    data = pd.read_csv(RETURN_FILE)

    if request.method == 'POST':
        try:
            # Ambil input tanggal dari form
            in_sample_start = request.form.get('in_sample_start')
            in_sample_end = request.form.get('in_sample_end')
            out_sample_start = request.form.get('out_sample_start')
            out_sample_end = request.form.get('out_sample_end')

            if not all([in_sample_start, in_sample_end, out_sample_start, out_sample_end]):
                return render_template('split_data.html', error="Semua kolom harus diisi.")

            # Konversi kolom Date ke datetime
            data['Date'] = pd.to_datetime(data['Date'])

            # Filter data berdasarkan tanggal
            in_sample_data = data[(data['Date'] >= in_sample_start) & (data['Date'] <= in_sample_end)]
            out_sample_data = data[(data['Date'] >= out_sample_start) & (data['Date'] <= out_sample_end)]

            # Simpan hasil split ke CSV
            in_sample_data.to_csv('in_sample_data.csv', index=False)
            out_sample_data.to_csv('out_sample_data.csv', index=False)

            # Ambil 5 baris pertama untuk ditampilkan di HTML
            in_sample_head = in_sample_data.head(5).to_dict(orient='records')
            out_sample_head = out_sample_data.head(5).to_dict(orient='records')

            return render_template(
                'split_data.html',
                in_sample_shape=in_sample_data.shape,
                out_sample_shape=out_sample_data.shape,
                in_sample_head=in_sample_head,
                out_sample_head=out_sample_head,
                success="Data telah berhasil di-split dan disimpan!"
            )

        except Exception as e:
            return render_template('split_data.html', error=f"Terjadi kesalahan: {e}")

    return render_template('split_data.html')


@app.route('/model_ar', methods=['GET', 'POST'])
def identifikasi_ar():
    # Pastikan file in_sample_data.csv tersedia
    if not os.path.exists('in_sample_data.csv'):
        return render_template('model_ar.html', error="File in_sample_data.csv tidak ditemukan.")

    # Load data in-sample
    data = pd.read_csv('in_sample_data.csv')
    data['Date'] = pd.to_datetime(data['Date'])

    if 'Return' not in data.columns:
        return render_template('model_ar.html', error="Kolom 'Return' tidak ditemukan dalam data.")

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
            acf_pacf_image=img_path
        )

    except Exception as e:
        return render_template('model_ar.html', error=f"Terjadi kesalahan: {e}")

@app.route('/asumsi_ar', methods=['GET', 'POST'])
def asumsi_klasik_ar():
    error = None
    ks_statistic = ks_pvalue = W_critical = None
    dw_stat = None
    lm_stat = lm_pvalue = t_critical_arch = None
    dw_lower, dw_upper = 1.5, 2.5  # Rentang kritis DW

    # Pastikan file in_sample_data.csv tersedia
    if not os.path.exists('in_sample_data.csv'):
        return render_template('asumsi_ar.html', error="File in_sample_data.csv tidak ditemukan.")

    # Load data
    data = pd.read_csv('in_sample_data.csv')
    
    if 'Return' not in data.columns:
        return render_template('asumsi_ar.html', error="Kolom 'Return' tidak ditemukan dalam data.")
    
    # Ambil input p dari user
    p = request.form.get("p")
    
    if p:
        try:
            p = int(p)
            if p < 1 or p > 10:  # Batasan nilai p
                raise ValueError("Nilai p harus antara 1 dan 10.")

            # Estimasi model AR dengan input p
            model_ar = sm.tsa.ARIMA(data['Return'], order=(p, 0, 0)).fit()
            residuals = model_ar.resid
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
        t_critical_arch=t_critical_arch
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
        return render_template('linearitas.html', error="File in_sample_data.csv tidak ditemukan.")

    # Load data
    data = pd.read_csv('in_sample_data.csv')

    if 'Return' not in data.columns:
        return render_template('linearitas.html', error="Kolom 'Return' tidak ditemukan dalam data.")

    # Buat tabel preview 5 baris pertama
    table_preview = data.head().to_html(classes="table table-bordered", index=False)

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
        table_preview=table_preview
    )

@app.route('/fungsi_transisi', methods=['GET', 'POST'])
def uji_fungsi_transisi():
    error = None
    summary_aux = None
    model_selected = ""
    table_preview = None
    keputusan_t = {}
    keputusan_p = {}

    if not os.path.exists('in_sample_data.csv'):
        return render_template('fungsi_transisi.html', error="File in_sample_data.csv tidak ditemukan.")

    data_model = pd.read_csv('in_sample_data.csv')

    if 'Return' not in data_model.columns:
        return render_template('fungsi_transisi.html', error="Kolom 'Return' tidak ditemukan dalam data.")

    table_preview = data_model.head().to_html(classes="table table-bordered", index=False)

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
        table_preview=table_preview,
        summary_aux=summary_aux,
        t_critical= t_critical,
        t_stat_1=t_stat_1, p_value_1=p_value_1,
        t_stat_2=t_stat_2, p_value_2=p_value_2,
        t_stat_3=t_stat_3, p_value_3=p_value_3,
        keputusan_t=keputusan_t,
        keputusan_p=keputusan_p,
        model_selected=model_selected
    )

@app.route('/estimasi_estar', methods=['GET', 'POST'])
def estimasi_estar():
    if request.method == 'POST':
        try:
            # 🛑 Pastikan input ada sebelum digunakan
            if 'p' not in request.form or 'd' not in request.form:
                return jsonify({'error': "Harap masukkan nilai p dan d!"})

            # Ambil nilai p dan d dari form user
            p = int(request.form['p'])
            d = int(request.form['d'])

            # 🛑 Validasi nilai p dan d
            if p <= 0 or d <= 0:
                return jsonify({'error': "Nilai p dan d harus lebih besar dari 0."})

            # Load dataset
            data_model = pd.read_csv('in_sample_data.csv')

            if 'Return' not in data_model.columns:
                return jsonify({'error': "Kolom 'Return' tidak ditemukan dalam data."})

            # **Step 1: Persiapan Data**
            for i in range(1, p+1):
                data_model[f'Lag{i}'] = data_model['Return'].shift(i)
            data_model['LagT'] = data_model['Return'].shift(d)
            data_model = data_model.dropna()

            # **Step 2: Definisi Model ESTAR**
            def ESTAR_transition(gamma, threshold, yt_lagged):
                return 1 - np.exp(-gamma * (yt_lagged - threshold) ** 2)

            def ESTAR_model(params, y, lags, yt_lagT):
                const_L = params[0]
                phiL = params[1:p+1]
                const_H = params[p+1]
                phiH = params[p+2:2*p+2]
                gamma = params[2*p+2]
                threshold = params[2*p+3]

                transition = ESTAR_transition(gamma, threshold, yt_lagT)
                yt_L = const_L + np.dot(lags, phiL)
                yt_H = const_H + np.dot(lags, phiH)

                return (1 - transition) * yt_L + transition * yt_H

            def nls_loss(params, y, lags, yt_lagT):
                y_hat = ESTAR_model(params, y, lags, yt_lagT)
                return np.sum((y - y_hat) ** 2)

            # **Step 3: Estimasi Model ESTAR**
            lags = data_model[[f'Lag{i}' for i in range(1, p+1)]]
            yt_lagT = data_model['LagT']
            y = data_model['Return']

            params_initial = np.random.uniform(-0.5, 0.5, size=2*p+4)
            result = minimize(nls_loss, params_initial, args=(y, lags, yt_lagT), method='BFGS')

            params_estimated = result.x
            hessian_inv = result.hess_inv
            standard_errors = np.sqrt(np.diag(hessian_inv))
            t_values = params_estimated / standard_errors
            p_values = 2 * (1 - t_dist.cdf(np.abs(t_values), df=len(data_model) - len(params_estimated)))

            # **Step 4: Buat DataFrame Hasil**
            param_names = ['Const_L'] + [f'phiL_{i}' for i in range(1, p+1)] + \
                          ['Const_H'] + [f'phiH_{i}' for i in range(1, p+1)] + ['gamma', 'threshold']

            results_df = pd.DataFrame({
                'Parameter': param_names,
                'Estimate': params_estimated,
                'Standard Error': standard_errors,
                't-value': t_values,
                'p-value': p_values
            })

            # Konversi hasil ke format HTML
            results_html = results_df.to_html(classes='table table-bordered')

            return render_template('estimasi_estar.html', tables=[results_html])

        except Exception as e:
            return jsonify({'error': str(e)})

    return render_template('estimasi_estar.html')

@app.route('/asumsi_estar', methods=['GET', 'POST'])
def asumsi_klasik_estar():
    results = None

    if request.method == 'POST':
        try:
            # Ambil input dari form
            p = int(request.form['p'])
            d = int(request.form['d'])

            # Simulasi dataset (Gantilah dengan dataset asli)
            data_model = pd.DataFrame({"Return": np.random.randn(100)})
            
            # Membuat lag variabel
            data_model['LagT'] = data_model['Return'].shift(d)
            for i in range(1, p + 1):
                data_model[f'Lag{i}'] = data_model['Return'].shift(i)
            data_model = data_model.dropna()

            # Simulasi parameter hasil estimasi (Gantilah dengan hasil model asli)
            params_estimated = np.random.randn(2 * p + 4)

            # Fungsi ESTAR Model (Placeholder, gantilah dengan model yang digunakan)
            def ESTAR_model(params, y, lags, lagT):
                return np.mean(y)  # Placeholder

            # Menghitung residual dari model ESTAR
            y_hat = ESTAR_model(params_estimated, data_model['Return'][p:], data_model.iloc[p:, 1:p+1], data_model['LagT'][p:])
            residuals = data_model['Return'][p:] - y_hat

            # 🔹 **1. Uji Durbin-Watson (Autokorelasi)**
            dw_stat = durbin_watson(residuals)

            # 🔹 **2. Uji ARCH-LM (Homoskedastisitas)**
            def arch_lm_test(residuals, m=1):
                residuals_sq = residuals ** 2
                SSR1 = np.sum(residuals_sq)
                omega_bar = np.mean(residuals_sq)
                SSR0 = np.sum((residuals_sq - omega_bar) ** 2)
                LM_stat = ((SSR0 - SSR1) / m) / (SSR1 / (len(residuals) - 2 * m - 1))
                p_value = 1 - chi2.cdf(LM_stat, m)
                return LM_stat, p_value

            arch_stat, arch_pval = arch_lm_test(residuals)

            # 🔹 **3. Uji Kolmogorov-Smirnov (Normalitas)**
            ks_stat, ks_pval = kstest(residuals, 'norm', args=(np.mean(residuals), np.std(residuals)))
            ks_critical = 1.36 / np.sqrt(len(residuals))

            # 🔹 Menyusun hasil uji asumsi
            results = {
                "Durbin-Watson": round(dw_stat, 4),
                "ARCH-LM Stat": round(arch_stat, 4),
                "ARCH-LM p-value": round(arch_pval, 4),
                "Kolmogorov-Smirnov Stat": round(ks_stat, 4),
                "Kolmogorov-Smirnov p-value": round(ks_pval, 4),
                "KS Critical Value": round(ks_critical, 4)
            }

        except Exception as e:
            results = {"Error": str(e)}

    return render_template('asumsi_estar.html', results=results)

@app.route('/prediksi_estar')
def prediksi_estar():
    return render_template('prediksi_estar.html')

@app.route('/var')
def var():
    return render_template('var.html')

@app.route('/interpretasi')
def interpretasi():
    return render_template('interpretasi.html')

# Main Program
if __name__ == '__main__':
    app.run(debug=True)