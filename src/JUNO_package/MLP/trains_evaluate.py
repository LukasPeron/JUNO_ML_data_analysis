import numpy as np
import torch
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def train_test_loop(train_loader, test_loader, model, criterion, optimizer, n_epochs=100):
    # Variables to store training and testing metrics
    train_losses = []
    test_losses = []

    min_train_loss = []
    min_test_loss = []

    avg_diff_E_train = []
    avg_diff_x_train = []
    avg_diff_y_train = []
    avg_diff_z_train = []

    std_diff_E_train = []
    std_diff_x_train = []
    std_diff_y_train = []
    std_diff_z_train = []

    avg_diff_E_test = []
    avg_diff_x_test = []
    avg_diff_y_test = []
    avg_diff_z_test = []

    std_diff_E_test = []
    std_diff_x_test = []
    std_diff_y_test = []
    std_diff_z_test = []

    # Training loop
    for epoch in range(n_epochs + 1):
        # Training phase
        model.train()
        epoch_loss = 0.0
        all_train_preds = []
        all_train_labels = []
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            all_train_preds.append(outputs)
            all_train_labels.append(labels)

        all_train_preds = torch.cat(all_train_preds).detach()
        all_train_labels = torch.cat(all_train_labels).detach()
        avg_train_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Evaluate on test set
        model.eval()
        with torch.no_grad():
            test_epoch_loss = 0.0
            all_test_preds = []
            all_test_labels = []
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                test_epoch_loss += loss.item()
                all_test_preds.append(outputs)
                all_test_labels.append(labels)
        avg_test_loss = test_epoch_loss / len(test_loader)
        test_losses.append(avg_test_loss)
        all_test_preds = torch.cat(all_test_preds).detach()
        all_test_labels = torch.cat(all_test_labels).detach()

        print(f"{epoch + 1}/{n_epochs + 1}, Train Loss: {avg_train_loss:.2e}")

        if epoch%10==0 :
            min_train_loss.append(avg_train_loss)
            min_test_loss.append(avg_test_loss)
            diff_E = (all_test_labels[:, 0]*100 - all_test_preds[:, 0]*100).cpu().numpy()
            avg_diff_E_test.append(np.mean(diff_E))
            std_diff_E_test.append(np.std(diff_E))

            diff_x = (all_test_labels[:, 1]*17015 - all_test_preds[:, 1]*17015).cpu().numpy()
            avg_diff_x_test.append(np.mean(diff_x))
            std_diff_x_test.append(np.std(diff_x))

            diff_y = (all_test_labels[:, 2]*17015 - all_test_preds[:, 2]*17015).cpu().numpy()
            avg_diff_y_test.append(np.mean(diff_y))
            std_diff_y_test.append(np.std(diff_y))

            diff_z = (all_test_labels[:, 3]*17015 - all_test_preds[:, 3]*17015).cpu().numpy()
            avg_diff_z_test.append(np.mean(diff_z))
            std_diff_z_test.append(np.std(diff_z))

            diff_E_train = (all_train_labels[:,0]*100 - all_train_preds[:,0]*100).cpu().numpy()
            avg_diff_E_train.append(np.mean(diff_E_train))
            std_diff_E_train.append(np.std(diff_E_train))

            diff_x_train = (all_train_labels[:,1]*17015 - all_train_preds[:,1]*17015).cpu().numpy()
            avg_diff_x_train.append(np.mean(diff_x_train))
            std_diff_x_train.append(np.std(diff_x_train))

            diff_y_train = (all_train_labels[:,2]*17015 - all_train_preds[:,2]*17015).cpu().numpy()
            avg_diff_y_train.append(np.mean(diff_y_train))
            std_diff_y_train.append(np.std(diff_y_train))

            diff_z_train = (all_train_labels[:,3]*17015 - all_train_preds[:,3]*17015).cpu().numpy()
            avg_diff_z_train.append(np.mean(diff_z_train))
            std_diff_z_train.append(np.std(diff_z_train))
    
    diff_E_test = (all_test_labels[:, 0]*100 - all_test_preds[:, 0]*100).cpu().numpy()
    diff_x_test = (all_test_labels[:, 1]*17015 - all_test_preds[:, 1]*17015).cpu().numpy()
    diff_y_test = (all_test_labels[:, 2]*17015 - all_test_preds[:, 2]*17015).cpu().numpy()
    diff_z_test = (all_test_labels[:, 3]*17015 - all_test_preds[:, 3]*17015).cpu().numpy()

    return (train_losses, test_losses, 
            min_train_loss, min_test_loss, 
            avg_diff_E_train, avg_diff_x_train, avg_diff_y_train, avg_diff_z_train, 
            std_diff_E_train, std_diff_x_train, std_diff_y_train, std_diff_z_train, 
            avg_diff_E_test, avg_diff_x_test, avg_diff_y_test, avg_diff_z_test, 
            std_diff_E_test, std_diff_x_test, std_diff_y_test, std_diff_z_test,
            diff_E_train, diff_x_train, diff_y_train, diff_z_train, 
            diff_E_test, diff_x_test, diff_y_test, diff_z_test)