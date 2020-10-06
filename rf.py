def randomForestInLoop(i):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import confusion_matrix, accuracy_score, plot_confusion_matrix
    from tensorflow.keras.datasets import mnist
    import numpy as np
    import pandas as pd
    (x_train_orig, y_train_all), (x_test_orig, y_test) = mnist.load_data()
    x_train_all = x_train_orig.reshape(60000, 784)
    x_test = x_test_orig.reshape(10000, 784)
    x_train = x_train_all[:54000]
    y_train = y_train_all[:54000]
    x_dev = x_train_all[54000:]
    y_dev = y_train_all[54000:]
    
    n = round(i/1000)   ## Use this to transform input XXXYYY into XXX YYY
    m = i - (n * 1000)  ## Use this to transform input XXXYYY into XXX YYY
    rfc = RandomForestClassifier(random_state=0, n_estimators = n, max_depth = m)
    rfc.fit(x_train, y_train)
    train_acc = round(accuracy_score(y_train, rfc.predict(x_train)), 4)
    dev_acc = round(accuracy_score(y_dev, rfc.predict(x_dev)), 4)
    return([n, m, train_acc, dev_acc])
