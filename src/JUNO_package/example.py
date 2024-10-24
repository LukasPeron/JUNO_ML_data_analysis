from MLP.MLP_model import *
from MLP.plots_results import *
from MLP.preprocess_data import *
from MLP.trains_evaluate import *

X, y = load_data()
X, y = scale_data(X, y)
n_epochs=10
train_loader, test_loader, model, criterion, optimizer = load_model(X, y)
train_losses, test_losses, min_train_loss, min_test_loss, avg_diff_E_train, avg_diff_x_train, avg_diff_y_train, avg_diff_z_train, std_diff_E_train, std_diff_x_train, std_diff_y_train, std_diff_z_train, avg_diff_E_test, avg_diff_x_test, avg_diff_y_test, avg_diff_z_test, std_diff_E_test, std_diff_x_test, std_diff_y_test, std_diff_z_test,diff_E_train, diff_x_train, diff_y_train, diff_z_train, diff_E_test, diff_x_test, diff_y_test, diff_z_test = train_test_loop(train_loader, test_loader, model, criterion, optimizer, n_epochs=n_epochs)
plot_all_figures(train_losses, test_losses, 
                min_train_loss, min_test_loss, 
                avg_diff_E_train, avg_diff_x_train, avg_diff_y_train, avg_diff_z_train, 
                std_diff_E_train, std_diff_x_train, std_diff_y_train, std_diff_z_train, 
                avg_diff_E_test, avg_diff_x_test, avg_diff_y_test, avg_diff_z_test, 
                std_diff_E_test, std_diff_x_test, std_diff_y_test, std_diff_z_test,
                diff_E_train, diff_x_train, diff_y_train, diff_z_train,
                diff_E_test, diff_x_test, diff_y_test, diff_z_test, n_epochs=n_epochs)