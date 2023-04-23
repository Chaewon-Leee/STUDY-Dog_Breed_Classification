import matplotlib.pyplot as plt
import os
import shutil
import numpy as np

def average_accuracy(result):
    accuracy = [ ]
    val_accuracy = [ ]
    for result_per_epoch in result:
        accuracy.append(result_per_epoch['accuracy'])
        val_accuracy.append(result_per_epoch['val_accuracy'])
    avg_acc = np.mean(accuracy)
    avg_val_acc = np.mean(val_accuracy)
    return avg_acc, avg_val_acc

def visualization_loss(result):
    loss = [ ]
    val_loss = [ ]
    epochs = [i+1 for i in range(len(result))]

    for result_per_epoch in result:
      loss.append(result_per_epoch['loss'])
      val_loss.append(result_per_epoch['val_loss'])

    plt.plot(epochs, loss, color='blue')
    plt.plot(epochs, val_loss, color='orange')

    plt.xlabel('Epoch')
    plt.ylabel("Loss")
    plt.legend(['loss', 'val_loss'])

    if os.path.exists('./result'):
      shutil.rmtree('./result')

    os.mkdir('./result')
    plt.savefig('./result/result.png')

def visualization_test_loss(loss):
    epochs = [i+1 for i in range(len(loss))]

    plt.plot(epochs, loss, color='blue')

    plt.xlabel('Epoch')
    plt.ylabel("Loss")
    plt.legend(['test loss'])

    if os.path.exists('./test_result'):
      shutil.rmtree('./test_result')

    os.mkdir('./test_result')
    plt.savefig('./test_result/result.png')
