"""
чей то код с гитхаба, скопировал для теста лабы
"""


def mix_columns(state):
    # Matrice de multiplication pour la transformation MixColumns dans l'algorithme AES
    matrix = [
        [0x02, 0x03, 0x01, 0x01],
        [0x01, 0x02, 0x03, 0x01],
        [0x01, 0x01, 0x02, 0x03],
        [0x03, 0x01, 0x01, 0x02]
    ]

    new_state = [[0 for _ in range(4)] for _ in range(4)]

    # Parcours des colonnes de l'état
    for col in range(4):
        # Parcours des lignes de l'état
        for row in range(4):
            val = 0
            # Multiplication et XOR pour chaque élément de la colonne
            for i in range(4):
                val ^= mult_bytes(matrix[row][i], state[i][col])
            new_state[row][col] = val

    return new_state


def mult_bytes(a, b):
    result = 0
    for _ in range(8):
        if b & 1 == 1:
            result ^= a
        hi_bit_set = a & 0x80
        a <<= 1
        if hi_bit_set == 0x80:
            a ^= 0x1b
        b >>= 1
    return result


# Exemple d'utilisation
state = [
    [180, 87, 196, 2],
    [216, 73, 179, 157],
    [151, 195, 255, 92],
    [35, 161, 105, 197]
]

new_state = mix_columns(state)
# Affichage du résultat en format hexadécimal
print("Le resultat en hexa est: ", [i for row in new_state for i in row])

# decimal
print("Le resultat eb decimal est: ", new_state)
