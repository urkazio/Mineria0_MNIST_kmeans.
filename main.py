import sys
import sklearn
import matplotlib
import numpy as np
import sns as sns
from keras.datasets import mnist
from matplotlib import pyplot as plt
from sklearn.cluster import MiniBatchKMeans
from sklearn import metrics
from sklearn.datasets import images
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy.spatial import distance
import sys


global array_conf
array_conf = []


# --------------------------------------------------------------------------------------------------
# ----------------------------------------- SUBPROGRAMAS  ------------------------------------------
# --------------------------------------------------------------------------------------------------


def infer_cluster_labels(kmeans, actual_labels):
    """
    un cluster se define por su contenido pero no sabemos cada cluster 'qué' es (en este caso qué numero representa)
    El metodo 'infer_cluster_labels' asocia la etiqueta mas probable a cada cluster
    :param kmeans: modelo
    :param actual_labels: la lista de etiquetas
    :return diccionario 'cluster:etiqueta' que contiene la etiqueta asociada/predicha para cada cluster
    """

    inferred_labels = {}

    for i in range(kmeans.n_clusters):

        # obtener las posibles etiquetas
        labels = []
        index = np.where(kmeans.labels_ == i)
        labels.append(actual_labels[index])

        # obtener cuál es la etiqueta más común en el cluster
        if len(labels[0]) == 1:
            counts = np.bincount(labels[0])
        else:
            counts = np.bincount(np.squeeze(labels))

        # añadir como etiqueta del cluster el numero que mas se repite en él
        if np.argmax(counts) in inferred_labels:
            inferred_labels[np.argmax(counts)].append(i)
        else:
            # si no existe el cluster en el diccionario de labels, crearlo.
            inferred_labels[np.argmax(counts)] = [i]

        arr = counts.tolist()
        array_conf.append(arr)

    return inferred_labels


def infer_data_labels(X_labels, cluster_labels):
    """
    Determina cual es la etiqueta para cada instancia teniendo en cuenta el cluster al que ha sido asociado
    Es decir, determina que todas las instancias dentro del cluster 'seis' son 'seises'
    :param X_labels: los cluster obtenidos tras la ejecucion de kmeans
    :param cluster_labels: diccionario que asocia cada cluster con su etiqueta
    :return array de etiquetas predichas para cada instancia (vector unidimensional ~= array)

    """

    # array vacio de length = 'numero de clusters'
    predicted_labels = np.zeros(len(X_labels)).astype(np.uint8)

    # recorre los cluster y obtiene el valor de la etiqueta asociada
    for i, cluster in enumerate(X_labels):
        for key, value in cluster_labels.items():
            if cluster in value:
                predicted_labels[i] = key

    return predicted_labels


def mostrar_matriz_confusion():
    ncluster = []
    prediciones = {"aciertos": 0, "errores": 0}

    print("\n--------------------------------------------------------------------")
    print(" --- Este es el número de instancias almacenada en cada cluster --- ")
    print("--------------------------------------------------------------------")
    unique, counts = np.unique(X_clusters, return_counts=True)
    print(dict(zip(unique, counts)))
    print("\n20 Etiquetas reales (train):")
    print(Y[:20])
    print("Predicción de dichas 20 etiquetas:")
    print(predicted_labels[:20])


    conf_matrix = []
    id = 0

    for cluster in array_conf:
        max = -1
        pos = 0
        pos_max = -1
        for apariciones in cluster:
            if (apariciones > max):
                max = apariciones
                pos_max = pos
                aux = cluster
            pos = pos + 1
        conf_matrix.insert(id, ["Cluster de " + str(pos_max) + "'s", aux, pos_max])
        id = id + 1

    conf_matrix.sort(key=lambda row: (row[0]), reverse=False)
    print(conf_matrix)

    # crear el heat-map

    nums = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
    nombre_clus = []
    apariciones_clus = []


    for row in conf_matrix:  #["Cluster de x"  [cont_instancias], instancia_max]
        numero_act = 0
        clusterAct = str(row[0])
        nombre_clus.append(clusterAct)
        apariciones = row[1]
        print(len(apariciones))
        apariciones_clus.append(apariciones)

        for elem in apariciones:
            if numero_act == row[2]:  # si el numero actual coincide con la etiqueta del clutser --> es acierto
                prediciones["aciertos"] = prediciones["aciertos"] + elem
            else:
                prediciones["errores"] = prediciones["errores"] + elem

            clusterAct = clusterAct + (str(elem) + "\t\t")
            numero_act = numero_act + 1


    apariciones_clus_np = np.asarray(apariciones_clus)


    print(apariciones_clus_np)
    print(apariciones_clus)


    fig, ax = plt.subplots()
    im = ax.imshow(apariciones_clus_np, cmap="viridis")
    cbar = ax.figure.colorbar(im, ax=ax, shrink=0.5)

    # Show all ticks and label them with the respective list entries
    ax.set_xticks(np.arange(len(nums)), labels=nums)
    ax.set_yticks(np.arange(len(nombre_clus)), labels=nombre_clus)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    for i in range(len(nombre_clus)):
        for j in range(len(nums)):
            text = ax.text(j, i, apariciones_clus_np[i, j],
                           ha="center", va="center", color="w")


    ax.set_title("Mapa de calor instancias cluster")
    fig.tight_layout()
    plt.show()


    error = (prediciones["errores"] / (prediciones["aciertos"] + prediciones["errores"]))
    print("\nEl indice de error es de : " + str(error))

    return error


# --------------------------------------------------------------------------------------------------
# -------------------------------------------- PRINCIPAL -------------------------------------------
# --------------------------------------------------------------------------------------------------


'''
 impotar la base de datos en los conjuntos test y train
 x --> imagenes
 y --> etiquetas
'''

(x_train, y_train), (x_test, y_test) = mnist.load_data()

print('Training Data: {}'.format(x_train.shape))
print('Training Labels: {}'.format(y_train.shape))
print('Testing Data: {}'.format(x_test.shape))
print('Testing Labels: {}'.format(y_test.shape))

# visualizar 16 instancias en un cuadro 4x4
fig, axs = plt.subplots(4, 4, figsize=(20, 20))
plt.gray()

# añadir las instancias a la figura junto con la etiqueta asociada
for i, ax in enumerate(axs.flat):
    ax.matshow(x_train[i])
    ax.axis('off')
    ax.set_title('Number {}'.format(y_train[i]))

# mostrar la figura
fig.show()

# ---------------- PREPROCESADO -----------------

# las imagenes guardadas como arrays numpy son de 2 dimensiones y kmeans solo soporta arrays de una dimension
# --> convertir cada imagen en un array de una dimensión:
X = x_train.reshape(len(x_train), -1)
X = X.astype(float) / 255.
print(len(X))
Y = y_train


# Coger una porcion para agilizar
X_s = X[0:10000]
Y_s = Y[0:10000]

errores = dict()
xpoints = []
ypoints = []


dimensiones_pca = np.arange(2,3)

for num_dim in dimensiones_pca:
    print(num_dim)
    array_conf = []

    '''
    # PCA
    X = StandardScaler().fit_transform(X_s)
    pca = PCA(n_components=15)
    X = pca.fit_transform(X)

    plt.scatter(X[:, 0], X[:, 1], s=5, c=Y_s, cmap='Spectral')
    plt.gca().set_aspect('equal', 'datalim')
    plt.colorbar(boundaries=np.arange(11) - 0.5).set_ticks(np.arange(10))
    plt.title('Visualización datos MNIST tras PCA = ' + str(num_dim), fontsize=10);
    plt.show()



    #TSNE
    X = TSNE(random_state = 42, n_components=num_dim,verbose=0, perplexity=40, n_iter=300).fit_transform(X)

    plt.scatter(X[:, 0], X[:, 1], s= 5, c=Y_s, cmap='Spectral')
    plt.gca().set_aspect('equal', 'datalim')
    plt.colorbar(boundaries=np.arange(11)-0.5).set_ticks(np.arange(10))
    plt.title('Visualización datos MNIST tras t-SNE', fontsize=20);
    plt.show()
    '''



    # ------------------- K-MEANS --------------------

    # inicializar el modelo con 10 clusters (números del 0-9)
    nclusters = 10
    kmeans = MiniBatchKMeans(n_clusters=nclusters)

    # ajustar el modelo a la porción de entrenamiento (train) -- entrenar el modelo
    kmeans.fit(X)
    # obtener las etiquetas provisionales de los cluster
    kmeans.labels_

    cluster_labels = infer_cluster_labels(kmeans, Y)  # predecir la etiqueta del cluster en base a su contenido
    X_clusters = kmeans.predict(X)
    predicted_labels = infer_data_labels(X_clusters, cluster_labels)

    errores[num_dim] = mostrar_matriz_confusion()

#f = plt.figure()
#f.set_figwidth(12)
#f.set_figheight(5)

for key in errores:
    xpoints.append(key);
    ypoints.append(errores[key])
    plt.plot(key,errores[key])

plt.plot(xpoints, ypoints)
plt.xticks(dimensiones_pca)
plt.title("Error total según la redimensión")
plt.xlabel("Número de dimensiones")
plt.ylabel("Error total")

plt.show()
print(errores)


# ------------------- VISUALIZACION DE LOS CENTROIDES --------------------

dict_item =dict()

centroides = kmeans.cluster_centers_  # obtener los centroides

# convertir los centroides en imagenes
imagenes = centroides.reshape(nclusters, 28, 28)  # 10 centroides de 28x28 dimensiones
imagenes *= 255
imagenes = imagenes.astype(np.uint8)

fig, axs = plt.subplots(1, nclusters, figsize=(nclusters*2.5, 5))
plt.gray()

# bucle por cada bloque de la grafica y añadir el centroide

for i, ax in enumerate(axs.flat):

    # establecer el label del cluster
    for key, value in cluster_labels.items():
        if i in value:
            ax.set_title('Inferred Label: {}'.format(key))
            dict_item[key] = value  #dict_item --> array:(etiqueta_Asociada, idx en "centroides[]")

    # añadir la imagen al bloque
    ax.matshow(imagenes[i])
    ax.axis('off')

# mostrar la figura
fig.show()



# ------------------- TESTEO  --------------------

X_test = x_test.reshape(len(x_test), -1)
X_test = X_test.astype(float) / 255.
print(len(X_test))

#X_test = pca.transform(X_test)

centroides = centroides.reshape(len(centroides), -1)
centroides = centroides.astype(float) / 255.


fil, col = 10, nclusters
array_conf = [[0 for x in range(fil)] for y in range(col)]
print(array_conf)

centroidTag = dict()

for key in dict_item:  #convertir la estructura (etiqueta,[id_centroide]) --> (id_centroide, etiqueta)
    etiqueta = key
    ids_centroides = dict_item[key]
    for id in ids_centroides:
        centroidTag[id] = etiqueta


for test_elem, test_tag in zip(X_test, y_test):  # recorrer las instancias de test junto con su etiqueta simultaneamente
    dst_min= sys.maxsize  # int mas grande en python3
    dst = sys.maxsize  # int mas grande en python3

    id = 0
    for centroide in centroides:
        #dst = np.linalg.norm(np.array(test_elem) - np.array(centroide))
        dst = distance.euclidean(test_elem, centroide)
        print("distancia "+str(dst))
        if dst < dst_min:
            dst_min = dst
            id_centroide = id
        id = id+1

    print("test(i) con etiqueta " + str(test_tag) + " ha sido asignado al cluster id " + str(id_centroide) + " cuya etiqueta es " + str([centroidTag[id_centroide]]))
    array_conf[id_centroide][test_tag] = array_conf[id_centroide][test_tag] + 1  # hacer el recuento de que se ha añadido dicha instancia

mostrar_matriz_confusion()





# saber dimensiones pca con el error -- no se cambia bien la dimension
# accuracy con pca y tsne (obetener el cambio de base y aplicarlo al test)
# evaluar el test por distancias -- se calcula mal la distancia