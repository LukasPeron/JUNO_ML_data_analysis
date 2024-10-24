import matplotlib
matplotlib.use('Agg')  # Use 'Agg' backend if running without display
import matplotlib.pyplot as plt

pwd_saving = "/pbs/home/l/lperon/work_JUNO/figures/MLP/"

def plot_all_figures(train_losses, test_losses, 
                     min_train_loss, min_test_loss, 
                     avg_diff_E_train, avg_diff_x_train, avg_diff_y_train, avg_diff_z_train, 
                     std_diff_E_train, std_diff_x_train, std_diff_y_train, std_diff_z_train, 
                     avg_diff_E_test, avg_diff_x_test, avg_diff_y_test, avg_diff_z_test, 
                     std_diff_E_test, std_diff_x_test, std_diff_y_test, std_diff_z_test,
                     diff_E_train, diff_x_train, diff_y_train, diff_z_train,
                     diff_E_test, diff_x_test, diff_y_test, diff_z_test,
                     batch_size=50, dropout_prop=0.2, lr=1e-5, n_epochs=100):

    # Plotting and visualization of results
    x = range(0,n_epochs+1,10)
    plt.figure(0)
    plt.plot(x, min_train_loss, "db", label="Train")
    plt.plot(x, min_test_loss, "dr", label="Test")
    plt.grid()
    plt.legend()
    plt.xlabel("Number of epochs")
    plt.ylabel("Last value of Loss")
    plt.savefig(pwd_saving+f"loss_vs_epochs_lr={lr:.0e}_batch_size={batch_size:.0f}_dropout_prop={dropout_prop:.1f}.svg")

    plt.figure(10)
    plt.loglog(x, min_train_loss, "db", label="Train")
    plt.loglog(x, min_test_loss, "dr", label="Test")
    plt.grid()
    plt.legend()
    plt.xlabel("Number of epochs")
    plt.ylabel("Last value of Loss")
    plt.savefig(pwd_saving+f"loss_vs_epochs_loglog_lr={lr:.0e}_batch_size={batch_size:.0f}_dropout_prop={dropout_prop:.1f}.svg")

    fig = plt.figure(1)
    ax = plt.axes()
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.plot(x, avg_diff_E_train, color="b", marker="o", linestyle="", label="Train")
    ax.plot(x, avg_diff_E_test, color="r", marker="o", linestyle="", label="Test")
    ax.set_xlabel("Number of epochs")
    ax.set_ylabel(r"$\langle E_{true}-E_{model} \rangle [MeV]$")
    plt.grid()
    plt.legend()
    plt.savefig(pwd_saving+f"diff_E_vs_epochs_lr={lr:.0e}_batch_size={batch_size:.0f}_dropout_prop={dropout_prop:.1f}.svg")

    fig = plt.figure(2)
    ax = plt.axes()
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.plot(x, avg_diff_x_train, color="b", marker="o", linestyle="", label="Train")
    ax.plot(x, avg_diff_x_test, color="r", marker="o", linestyle="", label="Test")
    ax.set_xlabel("Number of epochs")
    ax.set_ylabel(r"$\langle x_{true}-x_{model} \rangle [mm]$")
    plt.grid()
    plt.legend()
    plt.savefig(pwd_saving+f"diff_x_vs_epochs_lr={lr:.0e}_batch_size={batch_size:.0f}_dropout_prop={dropout_prop:.1f}.svg")

    fig = plt.figure(3)
    ax = plt.axes()
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.plot(x, avg_diff_y_train, color="b", marker="o", linestyle="", label="Train")
    ax.plot(x, avg_diff_y_test, color="r", marker="o", linestyle="", label="Test")
    ax.set_xlabel("Number of epochs")
    ax.set_ylabel(r"$\langle y_{true}-y_{model} \rangle [mm]$")
    plt.grid()
    plt.legend()
    plt.savefig(pwd_saving+f"diff_y_vs_epochs_lr={lr:.0e}_batch_size={batch_size:.0f}_dropout_prop={dropout_prop:.1f}.svg")

    fig = plt.figure(4)
    ax = plt.axes()
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.plot(x, avg_diff_z_train, color="b", marker="o", linestyle="", label="Train")
    ax.plot(x, avg_diff_z_test, color="r", marker="o", linestyle="", label="Test")
    ax.set_xlabel("Number of epochs")
    ax.set_ylabel(r"$\langle z_{true}-z_{model} \rangle [mm]$")
    plt.grid()
    plt.legend()
    plt.savefig(pwd_saving+f"diff_z_vs_epochs_lr={lr:.0e}_batch_size={batch_size:.0f}_dropout_prop={dropout_prop:.1f}.svg")

    fig = plt.figure(5)
    ax = plt.axes()
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.plot(x, std_diff_E_train, color="b", marker="o", linestyle="", label="Train")
    ax.plot(x, std_diff_E_test, color="r", marker="o", linestyle="", label="Test")
    ax.set_xlabel("Number of epochs")
    ax.set_ylabel(r"$std(E_{true}-E_{model}) [MeV]$")
    plt.grid()
    plt.legend()
    plt.savefig(pwd_saving+f"std_diff_E_vs_epochs_lr={lr:.0e}_batch_size={batch_size:.0f}_dropout_prop={dropout_prop:.1f}.svg")

    fig = plt.figure(6)
    ax = plt.axes()
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.plot(x, std_diff_x_train, color="b", marker="o", linestyle="", label="Train")
    ax.plot(x, std_diff_x_test, color="r", marker="o", linestyle="", label="Test")
    ax.set_xlabel("Number of epochs")
    ax.set_ylabel(r"$std(x_{true}-x_{model}) [mm]$")
    plt.grid()
    plt.legend()
    plt.savefig(pwd_saving+f"std_diff_x_vs_epochs_lr={lr:.0e}_batch_size={batch_size:.0f}_dropout_prop={dropout_prop:.1f}.svg")

    fig = plt.figure(7)
    ax = plt.axes()
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.plot(x, std_diff_y_train, color="b", marker="o", linestyle="", label="Train")
    ax.plot(x, std_diff_y_test, color="r", marker="o", linestyle="", label="Test")
    ax.set_xlabel("Number of epochs")
    ax.set_ylabel(r"$std(y_{true}-y_{model}) [mm]$")
    plt.grid()
    plt.legend()
    plt.savefig(pwd_saving+f"std_diff_y_vs_epochs_lr={lr:.0e}_batch_size={batch_size:.0f}_dropout_prop={dropout_prop:.1f}.svg")

    fig = plt.figure(8)
    ax = plt.axes()
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.plot(x, std_diff_z_train, color="b", marker="o", linestyle="", label="Train")
    ax.plot(x, std_diff_z_test, color="r", marker="o", linestyle="", label="Test")
    ax.set_xlabel("Number of epochs")
    ax.set_ylabel(r"$std(z_{true}-z_{model}) [mm]$")
    plt.grid()
    plt.legend()
    plt.savefig(pwd_saving+f"std_diff_z_vs_epochs_lr={lr:.0e}_batch_size={batch_size:.0f}_dropout_prop={dropout_prop:.1f}.svg")

    fig, ax1 = plt.subplots()

    color = 'tab:blue'
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss', color=color)
    ax1.loglog(range(n_epochs+1), train_losses, color=color, label='Training Loss')
    ax1.loglog(range(n_epochs+1), test_losses, color='tab:cyan', label='Test Loss')
    ax1.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()
    fig.legend(loc=(0.13,0.15))
    plt.savefig(pwd_saving+f"loss_long_lr={lr:.0e}_batch_size={batch_size:.0f}_dropout_prop={dropout_prop:.1f}.svg")

    # Plot histograms of the differences
    plt.figure(figsize=(10, 8))

    plt.subplot(2, 2, 1)
    plt.hist(diff_E_train, bins=50, color='red', alpha=0.7, density=True)
    plt.hist(diff_E_test, bins=50, alpha=0.7, density=True)
    plt.xlabel(r'$E_{true} - E_{model}$ [MeV]')

    plt.subplot(2, 2, 2)
    plt.hist(diff_x_train, bins=50, color='red', alpha=0.7, density=True)
    plt.hist(diff_x_test, bins=50, alpha=0.7, density=True)
    plt.xlabel(r'$x_{true} - x_{model}$ [mm]')

    plt.subplot(2, 2, 3)
    plt.hist(diff_y_train, bins=50, color='red', alpha=0.7, density=True)
    plt.hist(diff_y_test, bins=50, alpha=0.7, density=True)
    plt.xlabel(r'$y_{true} - y_{model}$ [mm]')

    plt.subplot(2, 2, 4)
    plt.hist(diff_z_train, bins=50, color='red', alpha=0.7, density=True)
    plt.hist(diff_z_test, bins=50, alpha=0.7, density=True)
    plt.xlabel(r'$z_{true} - z_{model}$ [mm]')

    plt.tight_layout()
    plt.savefig(pwd_saving+f"diff_long_lr={lr:.0e}_batch_size={batch_size:.0f}_dropout_prop={dropout_prop:.1f}.svg")