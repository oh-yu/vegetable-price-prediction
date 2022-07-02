def pipeline_rnn(train_loader, train, test, test_y, future=375, num_epochs=100):
    # Variable To Store Prediction
    preds = []
    train_losses = []
    test_losses = []
    
    # Instantiate Model, Optimizer, Criterion, EarlyStopping
    model = RNN(input_size = train.shape[2]).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-3)
    criterion = nn.MSELoss()
    early_stopping = EarlyStopping(patience=30)

    # Training & Test Loop
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        idx = None

        for idx, (batch_x, batch_y) in enumerate(train_loader):
            # Forward
            out = model(batch_x)
            loss = criterion(out, batch_y)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            
            # Update Params
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
            optimizer.step()
            
            # Add Training Loss
            running_loss += loss.item()
        train_losses.append(running_loss / idx)
        
        # Test
        with torch.no_grad():
            model.eval()
            pred_y = model.predict(train, test, future)
            pred_y = pred_y.reshape(-1)
            loss = criterion(pred_y, test_y)
            preds.append(pred_y)
            test_losses.append(loss.item())

        # Early Stopping
        early_stopping(loss.item())
        if early_stopping.early_stop:
            print(f"early stop at: {np.min(test_losses)}")
            loss = np.min(test_losses)
            pred_y = preds[np.argmin(test_losses)]
            break
    return pred_y, loss