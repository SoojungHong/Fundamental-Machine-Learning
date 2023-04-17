
if __name__ == '__main__':
    flags = FLAGS()

    # datasets, add augmentation to training set
    training_dataset = NCaltech101(flags.training_dataset, augmentation=True)
    validation_dataset = NCaltech101(flags.validation_dataset)

    # construct loader, handles data streaming to gpu
    training_loader = Loader(training_dataset, flags, device=flags.device)
    validation_loader = Loader(validation_dataset, flags, device=flags.device)

    # model, and put to device
    model = Classifier()
    model = model.to(flags.device)

    # optimizer and lr scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.5)

    writer = SummaryWriter(flags.log_dir)

    iteration = 0
    min_validation_loss = 1000

    for i in range(flags.num_epochs):
        sum_accuracy = 0
        sum_loss = 0
        model = model.eval()

        print(f"Validation step [{i:3d}/{flags.num_epochs:3d}]")
        for events, labels in tqdm.tqdm(validation_loader):

            with torch.no_grad():
                pred_labels, representation = model(events)
                loss, accuracy = cross_entropy_loss_and_accuracy(pred_labels, labels)

            sum_accuracy += accuracy
            sum_loss += loss

        validation_loss = sum_loss.item() / len(validation_loader)
        validation_accuracy = sum_accuracy.item() / len(validation_loader)

        writer.add_scalar("validation/accuracy", validation_accuracy, iteration)
        writer.add_scalar("validation/loss", validation_loss, iteration)

        # visualize representation
        representation_vizualization = create_image(representation)
        writer.add_image("validation/representation", representation_vizualization, iteration)

        print(f"Validation Loss {validation_loss:.4f}  Accuracy {validation_accuracy:.4f}")

        if validation_loss < min_validation_loss:
            min_validation_loss = validation_loss
            state_dict = model.state_dict()

            torch.save({
                "state_dict": state_dict,
                "min_val_loss": min_validation_loss,
                "iteration": iteration
            }, "log/model_best.pth")
            print("New best at ", validation_loss)

        if i % flags.save_every_n_epochs == 0:
            state_dict = model.state_dict()
            torch.save({
                "state_dict": state_dict,
                "min_val_loss": min_validation_loss,
                "iteration": iteration
            }, "log/checkpoint_%05d_%.4f.pth" % (iteration, min_validation_loss))

        sum_accuracy = 0
        sum_loss = 0

        model = model.train()
        print(f"Training step [{i:3d}/{flags.num_epochs:3d}]")
        for events, labels in tqdm.tqdm(training_loader):
            optimizer.zero_grad()

            pred_labels, representation = model(events)
            loss, accuracy = cross_entropy_loss_and_accuracy(pred_labels, labels)

            loss.backward()

            optimizer.step()

            sum_accuracy += accuracy
            sum_loss += loss

            iteration += 1

        if i % 10 == 9:
            lr_scheduler.step()

        training_loss = sum_loss.item() / len(training_loader)
        training_accuracy = sum_accuracy.item() / len(training_loader)
        print(f"Training Iteration {iteration:5d}  Loss {training_loss:.4f}  Accuracy {training_accuracy:.4f}")

        writer.add_scalar("training/accuracy", training_accuracy, iteration)
        writer.add_scalar("training/loss", training_loss, iteration)

        representation_vizualization = create_image(representation)
        writer.add_image("training/representation", representation_vizualization, iteration)
