def coefficients_sgd(train, l_rate, n_epoch):
	coef = [0.0 for i in range(len(train[0]))]
	for epoch in range(n_epoch):
		sum_error = 0
		for row in train:
			yhat = predict(row, coef)
			error = row[-1] - yhat
			sum_error += error**2
			coef[0] = coef[0] + l_rate * error * yhat * (1.0 - yhat)
			for i in range(len(row)-1):
				coef[i + 1] = coef[i + 1] + l_rate * error * yhat * (1.0 - yhat) * row[i]
		print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, l_rate, sum_error))
	return coef

def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None):
    """Treinar a rede neural usando mini-batch stochastic
    gradient descent. O `training_data` é uma lista de tuplas
        `(x, y)` representando as entradas de treinamento e as
        saídas. Os outros parâmetros não opcionais são
        auto-explicativos. Se `test_data` for fornecido, então a
        rede será avaliada em relação aos dados do teste após cada
        época e progresso parcial impresso. Isso é útil para
        acompanhar o progresso, mas retarda as coisas substancialmente."""

    training_data = list(training_data)
    n = len(training_data)

    if test_data:
        test_data = list(test_data)
        n_test = len(test_data)

    for j in range(epochs):
        random.shuffle(training_data)
        mini_batches = [training_data[k:k+mini_batch_size] for k in range(0, n, mini_batch_size)]
        
        for mini_batch in mini_batches:
            self.update_mini_batch(mini_batch, eta)
        
        if test_data:
            print("Epoch {} : {} / {}".format(j,self.evaluate(test_data),n_test));
        else:
            print("Epoch {} finalizada".format(j))