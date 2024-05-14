import matplotlib.pyplot as plt
from neural_network import NeuralNetwork
from Activation import Sigmoid
from Optimazer import *

def split_data(data, labels, train_ratio):
    indices = np.arange(len(data))
    np.random.seed(43)
    np.random.shuffle(indices)
    train_size = int(len(data) * train_ratio)
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]
    return np.array(data)[train_indices], np.array(data)[test_indices], np.array(labels)[train_indices], np.array(labels)[test_indices]


def read_data(archivo):
    with open(archivo) as file:
        lines = [line.strip() for line in file if line.strip()]

    matrices = []
    for i in range(0, len(lines), 7):
        matrix = [list(map(int, line.split())) for line in lines[i:i + 7]]
        matrices.append(matrix)
    return matrices


archive = "TP3-ej3-digitos.txt"

matrices = [np.array(matrix).flatten() for matrix in read_data(archive)]
expected_output = np.array([[1], [0], [1], [0], [1], [0], [1], [0], [1], [0]])
#output is 1 if it s a mutiply of 2 and -1 whereas ; the values are in [-1;1]

input_size = len(matrices[0])
hidden_size = 15
output_size = 1
epochs = 5000

def errors_training_learning_rate(opti):
    
    training_percentage = 1.0
    run_number = 5
    learning_rates = [0.1, 0.01, 0.001]
    plt.figure()  # Créer une nouvelle figure pour le graphique global
    for learning_rate in learning_rates:
        all_errors = []
        for run in range(1, run_number + 1):
            X_train, X_test, y_train, y_test = split_data(matrices, expected_output, training_percentage)
            if (opti == "Adam"):
                optimizer = AdamOptimizer(learning_rate, beta1=0.9, beta2=0.999, epsilon=0.01)
            else : 
                optimizer = GradientDescentOptimizer(learning_rate)
            model = NeuralNetwork([input_size, hidden_size, output_size], Sigmoid(), optimizer, verbose=False)
            errors = model.train(X_train, y_train, epochs)
            all_errors.append(errors)
        mean_errors = np.mean(all_errors, axis=0)
        plt.plot(np.arange(1, len(mean_errors) + 1), mean_errors, label='Learning Rate: {}'.format(learning_rate))

    plt.yscale('log')
    plt.xlabel('Iterations')
    plt.ylabel('Errors')
    plt.title('Evolution of Errors medium during depending several learning rate on 5 runs')
    plt.legend()
    plt.grid(True)
    plt.show()

def errors_test_training_rate(opti,archi):
    learning_rate = 0.1
    run_number = 5
    training_percentages = [0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80]

    mean_errors = []  # Pour stocker les erreurs moyennes pour chaque pourcentage d'entraînement

    for training_percentage in training_percentages:
        all_errors = []
        for run in range(1, run_number + 1):
            X_train, X_test, y_train, y_test = split_data(matrices, expected_output, training_percentage)
            if (opti == "Adam"):
                optimizer = AdamOptimizer(learning_rate, beta1=0.9, beta2=0.999, epsilon=0.01)
            else : 
                optimizer = GradientDescentOptimizer(learning_rate)
            model = NeuralNetwork(archi,  Sigmoid(),optimizer, verbose=False)
            _ = model.train(X_train, y_train, epochs)
            result = model.predict(X_test)
            errors = [(result[i] - y_test[i])**2 for i in range(len(result))]
            error = np.mean(errors)  # Calcul de l'erreur moyenne
            all_errors.append(error/((1-training_percentage)*10))
        mean_error = np.mean(all_errors)  # Calcul de l'erreur moyenne sur les 5 exécutions
        mean_errors.append(mean_error)

    # Tracer les barres
    plt.bar(np.arange(len(training_percentages)), mean_errors)
    plt.xlabel('Porcentaje de entrenamiento')
    plt.ylabel('Error medio')
    'Error medio '
    plt.title('Error medio según porcentaje de entrenamiento (Learning Rate: {})'.format(learning_rate))
    plt.xticks(np.arange(len(training_percentages)), [str(int(p * 100)) + '%' for p in training_percentages])
    plt.grid(True)
    plt.show()


def errors_training_optimizer(training_ratio,archi):
    learning_rate = 0.1
    run_number = 5
    training_percentage = training_ratio

    plt.figure()  # Créer une nouvelle figure pour le graphique global
    
    all_errors = []
    for run in range(1, run_number + 1):

        X_train, X_test, y_train, y_test = split_data(matrices, expected_output, training_percentage)
        optimizer = AdamOptimizer(learning_rate, beta1=0.9, beta2=0.999, epsilon=0.01)
        model = NeuralNetwork(archi, Sigmoid(),optimizer, verbose=False)
        errors = model.train(X_train, y_train, epochs)
        all_errors.append(errors)
    mean_errors = np.mean(all_errors, axis=0)
    plt.plot(np.arange(1, len(mean_errors) + 1), mean_errors, label='Adam')


    all_errors = []
    for run in range(1, run_number + 1):

        X_train, X_test, y_train, y_test = split_data(matrices, expected_output, training_percentage)
        optimizer = GradientDescentOptimizer(learning_rate)
        model = NeuralNetwork(archi, Sigmoid(),optimizer, verbose=False)
        errors = model.train(X_train, y_train, epochs)
        all_errors.append(errors)
    mean_errors = np.mean(all_errors, axis=0)
    plt.plot(np.arange(1, len(mean_errors) + 1), mean_errors, label='GradientDescent')

    plt.yscale('log')
    plt.xlabel('Época')
    plt.ylabel('Errors')
    plt.title('Evolución del error medio con {} del data set para entrenar'.format(training_percentage))
    plt.legend()
    plt.grid(True)
    plt.show()





errors_training_learning_rate("Gradient")
errors_training_learning_rate("Adam")
 
errors_training_optimizer(0.20,[input_size, hidden_size, output_size])
errors_training_optimizer(0.20,[input_size, hidden_size,15, output_size])
errors_training_optimizer(0.50,[input_size, hidden_size, output_size])
errors_training_optimizer(0.50,[input_size, hidden_size,15, output_size])
errors_training_optimizer(0.80,[input_size, hidden_size, output_size])
errors_training_optimizer(0.80,[input_size, hidden_size,15, output_size])

errors_test_training_rate("Gradient",[input_size, hidden_size, output_size])
errors_test_training_rate("Gradient",[input_size, hidden_size,15, output_size])
errors_test_training_rate("Adam",[input_size, hidden_size, output_size])
errors_test_training_rate("Adam",[input_size, hidden_size,15, output_size])
