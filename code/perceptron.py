def predicao(w, x, bias):
    ativacao = sum(w[k] * x[k] for k in range(len(w))) + bias
    if ativacao >= 0:
        return 1
    else:
        return 0


# Cada linha contém x1, x2 e a classe esperada.
dados = [
    [2.7810836, 2.550537003, 0],
    [1.465489372, 2.362125076, 0],
    [3.396561688, 4.400293529, 0],
    [1.38807019, 1.850220317, 0],
    [3.06407232, 3.005305973, 0],
    [7.627531214, 2.759262235, 1],
    [5.332441248, 2.088626775, 1],
    [6.922596716, 1.77106367, 1],
    [8.675418651, -0.242068655, 1],
    [7.673756466, 3.508563011, 1],
]

w = [0.0, 0.0]
bias = 0.0
taxa_aprendizado = 0.1
epocas = 5

for epoca in range(epocas):
    for linha in dados:
        x = linha[:2]
        y = linha[2]
        erro = y - predicao(w, x, bias)
        bias = bias + taxa_aprendizado * erro
        for k in range(len(w)):
            w[k] = w[k] + taxa_aprendizado * erro * x[k]

print('Peso=', w)
print('Bias:', bias)