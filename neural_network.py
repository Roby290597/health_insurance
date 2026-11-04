from torch import nn


# Definiere ein einfaches neuronales Netz für Regression
class InsuranceRegressor(nn.Module):
    def __init__(self, input_dim):
        super(InsuranceRegressor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 12),
            nn.ReLU(),
            nn.Linear(12, 8),
            nn.ReLU(),
            nn.Linear(8, 4),
            nn.ReLU(),
            nn.Linear(4, 1)
        )
    def forward(self, x):
        return self.net(x)


class InsuranceRegressorGELU(nn.Module):
    def __init__(self, input_dim):
        super(InsuranceRegressorGELU, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.GELU(),
            nn.Linear(16, 12),
            nn.GELU(),
            nn.Linear(12, 8),
            nn.GELU(),
            nn.Linear(8, 4),
            nn.ReLU(),
            nn.Linear(4, 1)
        )
    def forward(self, x):
        return self.net(x)



##
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr, spearmanr


def metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    pearson_corr, _ = pearsonr(y_true, y_pred)
    spearman_corr, _ = spearmanr(y_true, y_pred)
    
    return '''
        MSE : {mse:.4f},
        R^2 : {r2:.4f},
        Pearson Correlation : {pearson_corr:.4f},
        Spearman Correlation : {spearman_corr:.4f}
        '''.format(mse=mse, r2=r2, pearson_corr=pearson_corr, spearman_corr=spearman_corr)



def metrics_summary(matric_dic, pred, gt, method_name=""):
    pearson_corr, _ = pearsonr(gt, pred)
    spearman_corr, _ = spearmanr(gt, pred)
    r2 = r2_score(gt, pred)
    matric_dic[method_name] = [pearson_corr, spearman_corr, r2]
    return matric_dic