def intra_epoch(batched_dataset, optimizer, val_batched_dataset, epoch):
    mix_loss_print = 0.0
    total_loss = 0.0
    step = 0
    start_time = time.time()
    clip_norm = 1
    epoch = epoch

    train_loss_sum = []
    for i, (inputs, labels) in enumerate(batched_dataset):
        with tf.GradientTape() as tape:
            out_list, _, _, _ = model(inputs, training=True)
            loss = loss_tf(out_list, labels)
            train_loss_sum.append(loss.numpy())
            if (i + 1) % 200 == 0:
              print(f'Training loss for {epoch+1}/{i+1}th batch: {loss}, Mean Training loss: {np.mean(train_loss_sum)}')
        gradients = tape.gradient(loss, model.trainable_variables)
        gradients, _ = tf.clip_by_global_norm(gradients, clip_norm)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    val_loss_sum = []
    for i, (inputs, labels) in enumerate(val_batched_dataset):
        out_list, _, _, _ = model(inputs, training=False)
        val_loss = loss_tf(out_list, labels)
        val_loss_sum.append(val_loss.numpy())
    print(f'Validation loss for {epoch+1}/{i+1}th batch: {val_loss}, Mean Validation loss: {np.mean(val_loss_sum)}')

    sisdr_sum = []
    for i, (inputs, labels) in enumerate(test_batched_dataset):
        out_list, _, _, _ = model(inputs, training=False)
        sisdr = loss_tf(out_list, labels)
        sisdr_sum.append(sisdr.numpy())
    print(f'Test loss for {epoch+1}/{i+1}th batch: {sisdr}, Mean Test SISDR loss: {np.mean(sisdr_sum)}')

    return np.mean(train_loss_sum), np.mean(val_loss_sum)


optimizer = keras.optimizers.AdamW(learning_rate=1e-3)
dataset = batched_dataset
net = model
ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=optimizer, net=net)
manager = tf.train.CheckpointManager(ckpt, './tf_ckpts', max_to_keep=3)

dataset = batched_dataset
net = model
ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=optimizer, net=net)
manager = tf.train.CheckpointManager(ckpt, './tf_ckpts', max_to_keep=3)

max_epoch = 100
min_delta=0.001
patience = 2
factor = 0.75
min_lr = 0.0000001

def train_and_checkpoint(net, manager):
  best_val_loss = float('inf')
  epochs_since_improvement = 0
  ckpt.restore(manager.latest_checkpoint)
  if manager.latest_checkpoint:
    print("Restored from {}".format(manager.latest_checkpoint))
  else:
    print("Initializing from scratch.")
  for epoch in range(max_epoch):
    start = time.time()
    train_loss, val_loss = intra_epoch(batched_dataset, optimizer, val_batched_dataset, epoch)
    print(f"<<<END OF EPOCH {epoch+1} | Training Loss for {epoch+1}th epoch {train_loss} | Validation Loss for {epoch+1}th epoch {val_loss}>>>")
    if val_loss < best_val_loss - min_delta:
        print(f'best loss before rendering {best_val_loss} and val loss {val_loss}')
        best_val_loss = val_loss
        epochs_since_improvement = 0
        print('epochs_since_improvement in best loss', epochs_since_improvement)
    else:
        epochs_since_improvement += 1
        print('epochs_since_improvement in not loss', epochs_since_improvement)
        if epochs_since_improvement >= patience:
            new_lr = optimizer.learning_rate * factor
            if new_lr >= min_lr:
                optimizer.learning_rate.assign(new_lr)
                print(f"Validation loss did not improve for {patience} epochs. Halving learning rate to {new_lr}")
                epochs_since_improvement = 0 
    ckpt.step.assign_add(1)
    if int(ckpt.step) % 1 == 0:
      save_path = manager.save()
      print("Saved checkpoint for step {}: {}".format(int(ckpt.step), save_path))

#run training
train_and_checkpoint(net, manager)
#model save
modep_for_pred.save("modep_for_pred.keras")
#model load
loaded_model_v3 = load_model("modep_for_pred.keras")
#model performance score
sisnri_sum = []
for i, (inputs, labels) in enumerate(test_batched_dataset):
    out_list, _, _, _ = loaded_model_v3(inputs, training=False)
    sisnr = SISNRi_tf().compute_loss(inputs, out_list, labels)
    sisnri_sum.append(sisnr.numpy())
    print(f'Test loss for {i+1}th file: {sisnr}, Mean Test SI-SNRi(dB) loss: {np.mean(sisnri_sum)}')
print('test dataset score in SI-SNRi(dB) is', {np.mean(sisnri_sum)})
