# import moment

# from apps.stocking.prediction import predict

# from sklearn.svm import SVC
# from sklearn.neural_network import MLPClassifier

# infos = {
#     'name': 'train-test-1',
#     'code': 'sz002415',
#     'timestamp': moment.now().format('YYYYMMDD_HHmmss'),
#     'features': [
#         'netamount_chg_1', 'ratioamount_chg_1', 'r0_net_chg_1', 'r0_ratio_chg_1',
#         'r0x_ratio__4', 'close_chg_1', 'high_chg_1', 'high_chg_2', 'high_chg_3',
#         'low_chg_1', 'low_chg_2', 'low_chg_3', 'open_chg_3', 'yest_chg_3',
#         'turnover__3', 'volumn', 'amount', 'circul_chg_10', 'volumn__4', 'low_chg_1',
#         'low_chg_2'
#     ],
#     'past_days': 5,
#     'future_days': 10,
#     'test_size': 0.05,
#     'random_state': 39,
#     'feature_pows': [2, 3, 4],
#     'feature_chgs': [1, 2, 3, 5, 10],
#     'zeros_copy_days': 10,
#     'classifier': 'SVC'
# }

# if __name__ == "__main__":
#     predict(infos, estimator=SVC(gamma=4, C=2))
