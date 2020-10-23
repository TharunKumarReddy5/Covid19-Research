from datetime import *
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import plot, savefig, legend, xlabel, show
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from numpy import zeros, linspace
from sklearn.metrics import *


def data_split(data, orders):
    x_train = np.empty((len(data) - orders, orders))
    y_train = data[orders:]
    for i in range(len(data) - orders):
        x_train[i] = data[i: orders + i]
    return x_train, y_train


def ridge(x, y):
    print('\nStart searching good parameters for the task..')
    parameters = {'alpha': np.arange(0, 0.100005, 0.000005).tolist(),
                  "tol": [1e-8],
                  'fit_intercept': [True, False],
                  'normalize': [True, False]}
    clf = GridSearchCV(Ridge(), parameters, n_jobs=-1, cv=5)
    clf.fit(x, y)
    print('\n Results for the parameters grid search :')
    print('Model : ', clf.best_estimator_)
    print('Score :', clf.best_score_)
    return clf


frame_main = pd.read_csv(r'F:\Goldmansachs\Datathon\India Data\State_dataset.csv')
reg = list(np.unique(frame_main['Region']))
union_terr = ['Dadra & Nagar Haveli and Daman & Diu', 'Lakshadweep', 'Ladakh', 'Puducherry']

for i in union_terr:
    # print(i)
    reg.remove(i)

reg = ['India']

for m in range(len(reg)):
    frame = frame_main[frame_main['Region'] == reg[m]]
    frame = frame[frame['Date']<='2020-09-14']
    set_size = len(frame)
    train_size = 164
    test_size = set_size - train_size
    frame1 = frame.head(train_size)
    X_tot = frame['totalConfirmed'] - frame['totalRecovered'] - frame['totalFatalities']
    R_tot = frame['totalRecovered'] + frame['totalFatalities']
    X = frame1['totalConfirmed'] - frame1['totalRecovered'] - frame1['totalFatalities']
    R = frame1['totalRecovered'] + frame1['totalFatalities']
    X = X.to_numpy()
    X_tot = X_tot.to_numpy()
    R = R.to_numpy()
    R_tot = R_tot.to_numpy()
    population = int(np.unique(frame['Population'])[0])
    S = population - X - R
    S_tot = population - X_tot - R_tot

    X_diff = np.array([X[:-1], X[1:]], dtype=np.float64).T
    R_diff = np.array([R[:-1], R[1:]], dtype=np.float64).T

    gamma = (R[1:] - R[:-1]) / X[:-1]
    beta = population * (X[1:] - X[:-1] + R[1:] - R[:-1]) / (X[:-1] * (population - X[:-1] - R[:-1]))

    for i in range(len(beta)):
        if beta[i] == np.nan or beta[i] == np.inf:
            beta[i] = beta[i - 1]

    for i in range(len(gamma)):
        if gamma[i] == np.nan or gamma[i] == np.inf:
            gamma[i] = gamma[i - 1]

    beta = np.nan_to_num(beta)
    gamma = np.nan_to_num(gamma)

    R0 = np.divide(beta, gamma, out=np.zeros_like(beta), where=gamma != 0)

    R0 = np.nan_to_num(R0)
    orders_beta = 30
    orders_gamma = 30
    start_beta = 10
    start_gamma = 10

    print("\nThe latest transmission rate beta of SIR model:", beta[-1])
    print("The latest recovering rate gamma for SIR model : ", gamma[-1])
    print("The latest reproduction number R0 :", R0[-1])

    ### SPLITING DATA INTO TRAINING SET AND TEST SET###

    x_beta, y_beta = data_split(beta, orders_beta)
    x_beta = np.nan_to_num(x_beta)
    y_beta = np.nan_to_num(y_beta)
    x_gamma, y_gamma = data_split(gamma, orders_gamma)

    ################### TRAINING #####################

    clf_beta = Ridge(alpha=0.0003675, copy_X=True, fit_intercept=False, max_iter=None, normalize=True,
                     random_state=None, solver='auto', tol=1e-08).fit(x_beta, y_beta)
    clf_gamma = Ridge(alpha=0.0001675, copy_X=True, fit_intercept=False, max_iter=None, normalize=True,
                     random_state=None, solver='auto', tol=1e-08).fit(x_gamma, y_gamma)

    # Predicting values of beta and gamma
    beta_hat = clf_beta.predict(x_beta)
    gamma_hat = clf_gamma.predict(x_gamma)

    print("\n r2 score for beta ", r2_score(y_beta, beta_hat))

    # Accuracy score for predicted beta and gamma
    print("R-squared score for Beta ", clf_beta.score(x_beta, y_beta))
    print("R-squared score for Gamma ", clf_gamma.score(x_gamma, y_gamma))

    # Plot actual beta vs predicted beta
    plt.figure(1)
    plt.plot(y_beta, label=r'$\beta (t)$')
    plt.plot(beta_hat, label=r'$\hat{\beta}(t)$')
    plt.legend()
    plt.show()

    # Plot actual gamma vs predicted gammma
    plt.figure(2)
    plt.plot(y_gamma, label=r'$\gamma (t)$')
    plt.plot(gamma_hat, label=r'$\hat{\gamma}(t)$')
    plt.legend()
    plt.show()

    ### Parameters for Time-dependent SIR Model (TSIR) ###########

    stop_X = 0  # stopping criteria
    stop_day = 120  # maximum iteration days

    day_count = 0
    turning_point = 0


    S_predict = [S[-1]]
    X_predict = [X[-1]]
    R_predict = [R[-1]]

    ### Predicting beta and gamma for future values with known values
    predict_beta = np.array(beta[-orders_beta:]).tolist()
    predict_gamma = np.array(gamma[-orders_gamma:]).tolist()

    while (X_predict[-1] >= stop_X) and (day_count < stop_day):
        if predict_beta[-1] > predict_gamma[-1]:
            turning_point += 1

        next_beta = clf_beta.predict(np.asarray([predict_beta[-orders_beta:]]))[0]
        next_gamma = clf_gamma.predict(np.asarray([predict_gamma[-orders_gamma:]]))[0]

        if next_beta < 0:
            next_beta = 0

        if next_gamma < 0:
            next_gamma = 0

        predict_beta.append(next_beta)
        predict_gamma.append(next_gamma)

        next_S = ((-predict_beta[-1] * S_predict[-1] * X_predict[-1]) / population) + S_predict[-1]

        next_X = ((predict_beta[-1] * S_predict[-1] * X_predict[-1]) / population) - (
                    predict_gamma[-1] * X_predict[-1]) + X_predict[-1]

        next_R = (predict_gamma[-1] * X_predict[-1]) + R_predict[-1]

        S_predict.append(next_S)
        X_predict.append(next_X)
        R_predict.append(next_R)

        day_count += 1

    #### PRINT STATS ########

    print('\n Newly Confirmed cases tomorrow :', np.rint(X_predict[1] + R_predict[1] - (X_predict[0] + R_predict[0])))
    print('Infected people tomorrow :', np.rint(X_predict[1]))
    print('Newly Recovered + Death toll tomorrow : ', np.rint(R_predict[1] - R_predict[0]))
    print('Confirmed cases on the end day :', np.rint(X_predict[-2] + R_predict[-2]))

    ####### Plot the time evolution of TSIR model ######

    plt.figure(3)
    plt.plot(range(len(X) - 1, len(X) - 1 + len(X_predict)), X_predict, '*-', label=r'$\hat{X}(t)$', color='darkorange')
    plt.plot(range(len(X) - 1, len(X) - 1 + len(X_predict)), R_predict, '*-', label=r'$\hat{R}(t)$', color='limegreen')
    plt.plot(range(len(X)), X, 'o--', label=r'$X(t)$', color='chocolate')
    plt.plot(range(len(X)), R, 'o--', label=r'$R(t)$', color='darkgreen')
    plt.xlabel('Day')
    plt.ylabel('Population')
    plt.title('Time evolution of the time-dependent SIR model (TSIR)')
    plt.legend()
    plt.show()

    #### Validation of TSIR model ########

    X1 = np.array(X_tot[-test_size:])
    R1 = np.array(R_tot[-test_size:])

    t = np.array(range(0, test_size))

    X_testpred = np.array(X_predict[:test_size])
    R_testpred = np.array(R_predict[:test_size])

    print("R squared core for R_actual ", r2_score(R1, R_testpred))

    plt.figure(4)
    plt.plot(t, X_testpred, '*-', label=r'$\hat{X}(t)$', color='darkorange')
    plt.plot(t, R_testpred, '*-', label=r'$\hat{R}(t)$', color='limegreen')
    plt.plot(t, X1, 'o--', label=r'$X(t)$', color='chocolate')
    plt.plot(t, R1, 'o--', label=r'$R(t)$', color='darkgreen')
    plt.xlabel('Day')
    plt.ylabel('Person')
    plt.title('One-day predictions of TSIR model')
    plt.legend()
    plt.show()

    error_X = []
    error_R = []
    R0 = []
    predicted_R0 = []
    error_R0 = []

    for i in range(len(X1)):
        error_X.append(abs(X1[i] - X_testpred[i]) / X1[i])

    for i in range(len(R1)):
        error_R.append(abs(R1[i] - R_testpred[i]) / R1[i])

    for i in range(len(beta)):
        if gamma[i] == 0:
            R0.append(beta[i] / gamma[i - 1])
        else:
            R0.append(beta[i] / gamma[i])

    for i in range(len(beta_hat)):
        predicted_R0.append(beta_hat[i] / gamma_hat[i])

    for i in range(len(predicted_R0)):
        error_R0.append(abs(R0[orders_beta + i] - predicted_R0[i]) / R0[orders_beta + i])

    plt.figure(5)
    plt.plot(t, error_X, 'o--', label=r'$X(t)$', color='chocolate')
    plt.plot(t, error_R, 'o--', label=r'$R(t)$', color='darkgreen')
    plt.xlabel('Day')
    plt.ylabel('Percentage Error')
    plt.title('Error rate in model')
    plt.legend()
    plt.show()

    err = np.mean(error_X)
    print('Mean Error % in Infected prediction is: ' + str(err * 100))
    mean_sqerr = mean_squared_error(np.array(X1) / np.array(X1), np.array(X_testpred) / np.array(X1))
    print('Mean squared error % for infected predictions is: ' + str(mean_sqerr * 100))

    t = np.array(range(orders_beta, train_size - 1))

    plt.figure(6)
    plt.plot(t, R0[orders_beta:], 'o--', label='actual_R0', color='chocolate')
    plt.plot(range(orders_beta - 1, t[-1]), predicted_R0, 'o--', label='predicted_R0', color='darkgreen')
    plt.xlabel('Day')
    plt.ylabel('R0')
    plt.title('R0 of the model')
    plt.legend()
    plt.show()

    plt.figure(7)
    plt.plot(error_R0, 'o--', label='R0_error', color='chocolate')
    plt.xlabel('Day')
    plt.ylabel('Percentage Error')
    plt.title('Error in R0 of the model')
    plt.legend()
    plt.show()

    ################ VACCINATION ################

    vacc_release = 20  # Vaccine release x days from now

    vacc_betatest = predict_beta[-(stop_day - test_size - vacc_release):]
    vacc_gammatest = predict_gamma[-(stop_day - test_size - vacc_release):]

    dt = 1  # 1 day
    D = stop_day - test_size - vacc_release  # simulate for D days
    N = int(D / dt)  # corresponding no of hours


    # Vaccination campaign

    def vacc_model(vacc_rate):
        Sv = zeros(N + 1)
        V = zeros(N + 1)
        I = zeros(N + 1)
        Rv = zeros(N + 1)
        R0v = zeros(N)
        p = zeros(N + 1)
        p = p + vacc_rate
        # Initial condition
        Sv[0] = S_predict[-(stop_day - test_size - vacc_release)]
        V[0] = 0
        I[0] = X_predict[-(stop_day - test_size - vacc_release)]
        Rv[0] = R_predict[-(stop_day - test_size - vacc_release)]
        r = 0
        revised_beta = 0
        revised_gamma = 0

        # Step equations forward in time
        for n in range(N):
            if n == 0:
                vacc_beta, vacc_gamma = (predict_beta[-(stop_day - test_size - vacc_release)],
                                         predict_gamma[-(stop_day - test_size - vacc_release)])
            else:
                revised_gamma = (Rv[n] - Rv[n - 1]) / I[n - 1]
                revised_beta = population * (I[n] - I[n - 1] + Rv[n] - Rv[n - 1]) / (
                            I[n - 1] * (population - I[n - 1] - Rv[n - 1]))
                vacc_beta, vacc_gamma = (revised_beta, revised_gamma)
                r = revised_beta / revised_gamma
            Sv[n + 1] = Sv[n] - (dt * vacc_beta * S[n] * I[n] / population) - dt * p[n] * Sv[n]
            V[n + 1] = V[n] + dt * p[n] * Sv[n]
            I[n + 1] = I[n] + (dt * vacc_beta * Sv[n] * I[n] / population) - dt * vacc_gamma * I[n]
            Rv[n + 1] = Rv[n] + dt * vacc_gamma * I[n]
            R0v = np.append(R0v, r)

        I_tot = np.append(np.append(X, X_predict[1:len(X_predict) - D - 1]), I)
        R_tot = np.append(np.append(R, R_predict[1:len(R_predict) - D - 1]), Rv)
        S_tot = np.append(np.append(S, S_predict[1:len(S_predict) - D - 1]), Sv)
        V_tot = np.append(zeros(len(S_tot) - D - 1), V)
        return (I_tot, R_tot, S_tot, V_tot)


    def vaccrate_change(rate):
        I_total = pd.DataFrame()
        R_total = pd.DataFrame()
        S_total = pd.DataFrame()
        V_total = pd.DataFrame()
        for i in range(len(rates)):
            iv, r, s, v = vacc_model(rates[i])
            I_total[i] = iv
            R_total[i] = r
            S_total[i] = s
            V_total[i] = v
        return I_total, R_total, S_total, V_total


    def plt_I(I_total, t):
        plt.figure(8)
        for i in range(len(rates)):
            pt = ['-', '--', '-.', ':'][i % 4]
            plt.plot(I_total[i], label='Ivac for p = ' + str(rates[i]), linestyle=pt)
        plt.plot(t, np.append(X, X_predict[1:]), marker="*", markersize=3, label='I_predict', color='tomato')
        plt.xlabel('Days')
        plt.ylabel('Population')
        plt.title('I change with vacc rate p')
        plt.legend(loc=3, bbox_to_anchor=(1, 0))
        plt.show()


    def plt_R(R_total, t):
        plt.figure(9)
        for i in range(len(rates)):
            pt = ['-', '--', '-.', ':'][i % 4]
            plt.plot(R_total[i], label='Rvac for p = ' + str(rates[i]), linestyle=pt)
        plt.plot(t, np.append(R, R_predict[1:]), marker="*", markersize=3, label='R_predict', color='mediumpurple')
        plt.xlabel('Days')
        plt.ylabel('Population')
        plt.title('R change with vacc rate p')
        plt.legend(loc=3, bbox_to_anchor=(1, 0))
        plt.show()


    def plt_IR(I_total, R_total, t):
        plt.figure(10)
        for i in range(len(rates)):
            pt = ['-', '--', '-.', ':'][i % 4]
            plt.plot(I_total[i], label='Ivac for p = ' + str(rates[i]), linestyle=pt)
            plt.plot(R_total[i], label='Rvac for p = ' + str(rates[i]), linestyle=pt)
        plt.plot(t, np.append(X, X_predict[1:]), marker="*", markersize=3, label='I_predict', color='tomato')
        plt.plot(t, np.append(R, R_predict[1:]), marker="*", markersize=3, label='R_predict', color='mediumpurple')
        plt.xlabel('Days')
        plt.ylabel('Population')
        plt.title('SIRV Compartmental Graph - I and R change wrt p')
        plt.legend(loc=3, bbox_to_anchor=(1, 0))
        plt.show()


    def plt_SV(S_total, V_total, t):
        plt.figure(11)
        for i in range(len(rates)):
            pt = ['-', '--', '-.', ':'][i % 4]
            plt.plot(S_total[i], label='Svac for p = ' + str(rates[i]), linestyle=pt, color='darkgoldenrod')
            plt.plot(V_total[i], label='Rvac for p = ' + str(rates[i]), linestyle=pt, color='darkgreen')
        plt.xlabel('Days')
        plt.ylabel('Population')
        plt.title('SIRV Compartmental Graph - S and V change wrt p')
        plt.legend(loc=3, bbox_to_anchor=(1, 0))
        plt.show()


    if __name__ == "__main__":
        rates = [0.005, 0.007, 0.01, 0.05]
        I_total, R_total, S_total, V_total = vaccrate_change(rates)
        t = linspace(0, len(S_total[0]), len(S_total[0]))
        plt_I(I_total, t)
        plt_R(R_total, t)
        plt_IR(I_total, R_total, t)
        plt_SV(S_total, V_total, t)
        I_total.columns = ['I_p_0.005', 'I_p_0.007', 'I_p_0.01', 'I_p_0.05']
        R_total.columns = ['R_p_0.005', 'R_p_0.007', 'R_p_0.01', 'R_p_0.05']
        S_total.columns = ['S_p_0.005', 'S_p_0.007', 'S_p_0.01', 'S_p_0.05']
        V_total.columns = ['V_p_0.005', 'V_p_0.007', 'V_p_0.01', 'V_p_0.05']
        I_total['I_predict'] = np.append(X, X_predict[1:])
        R_total['R_predict'] = np.append(R, R_predict[1:])
        main_data = pd.DataFrame()
        main_data = pd.concat([I_total, R_total, S_total, V_total], axis=1)
        l1 = list(beta)[:orders_beta] + list(beta_hat)[:-orders_beta] + list(predict_beta)
        l1.insert(0, 0)
        l2 = list(gamma)[:orders_gamma] + list(gamma_hat)[:-orders_gamma] + list(predict_gamma)
        l2.insert(0, 0)
        main_data['beta'] = l1
        main_data['gamma'] = l2
        main_data['Region'] = np.unique(frame['Region'])[0]
        main_data['Population'] = np.unique(frame['Population'])[0]
        main_data['Date'] = pd.date_range(start=np.array(frame['Date'])[0], periods=len(main_data), freq='D')
        main_data['I_actual'] = list(X_tot) + list(zeros(len(I_total) - len(X_tot)))
        main_data['R_actual'] = list(R_tot) + list(zeros(len(R_total) - len(R_tot)))
        main_data['S_actual'] = list(R_tot) + list(zeros(len(S_total) - len(S_tot)))  # typo or actually like this?
        main_data = pd.concat([main_data, frame[
            ['Confirmed', 'Deceased', 'Recovered', 'totalConfirmed', 'totalFatalities',
             'totalRecovered']].reset_index()], axis=1)
    if m == 0:
        main_data.to_csv(r'F:\Goldmansachs\Datathon\India Data\Tableau SIR Data.csv')
    else:
        main_data.to_csv(r'F:\Goldmansachs\Datathon\India Data\Tableau SIR Data.csv', mode='a', header=False)
    print(reg[m])