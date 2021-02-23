import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


"""Ανάγνωση δεδομένων από αρχείο csv"""
column_names = ["Sepal_Length","Sepal_Width","Petal_Length","Petal_Width","Class"]
iris_data = pd.read_csv('data/iris.data',names=column_names)

def xyz(iris_class):
    """Επιλέγουμε τα χαρακτηριστικά που μας ενδιαφέρουν ανάλογα με το είδος (κλάση) του φυτού και τα επιστρέφουμε ξεχωριστά ως pandas series."""
    x = iris_data[ iris_data.Class == iris_class ].Petal_Width
    y = iris_data[ iris_data.Class == iris_class ].Petal_Length
    z = iris_data[ iris_data.Class == iris_class ].Sepal_Length
    return x,y,z

def centroid_3d(x,y,z):
    """Βρίσκουμε το κέντρο συμμετρίας ενός σμήνους σημείων. Το σημείο αυτό ελαχιστοποιεί το άθροισμα των ευκλείδιων αποστάσεων μεταξύ αυτού και όλων των άλλων σημείων του σμήνους σε σχέση με οποιοδήποτε άλλο σημείο."""
    xc = np.sum(np.array(x))/len(x)
    yc = np.sum(np.array(y))/len(y)
    zc = np.sum(np.array(z))/len(z)
    return xc , yc , zc

def max_distance_from_centroid(x,y,z):
    """Βρίσκουμε τη μεγαλύτερη απόσταση μεταξύ κέντρου συμμετρίας και οποιουδήποτε άλλου σημείου στο σμήνος ώστε να την θέσουμε ως ακτίνα της σφαίρας."""
    x=np.array(x)
    y=np.array(y)
    z=np.array(z)
    cntr = np.array(centroid_3d(x,y,z))
    coord = np.array( [ [x[i],y[i],z[i]] for i in range(len(x)) ] )
    coord = np.power(coord-cntr,2)
    return np.max(np.sum(coord,axis=1)**(1/2))


def sphere(x,y,z,xc,yc,zc,clr):
    """Δημιουργούμε την σφαίρα εκείνη στην οποία θα περικλείονται όλα τα σημεία του εκάστοτε σμήνους σημείων."""
    u , v = np.mgrid[0:2*np.pi:18j, 0:np.pi:18j]
    R = max_distance_from_centroid(x,y,z)
    x = R*np.cos(u)*np.sin(v)+xc
    y = R*np.sin(u)*np.sin(v)+yc
    z = R*np.cos(v)+zc
    ax.plot_wireframe(x, y, z, color=clr)

#Συλλέγουμε τις στήλες δεδομένων για κάθε είδος φυτού που μας ενδιαφέρει
x1 , y1 , z1 = xyz("Iris-setosa")
#Βρίσκουμε το κέντρο συμμετρίας για το σμήνος σημείων του παραπάνω είδους φυτού.
x1c , y1c , z1c = centroid_3d(x1,y1,z1)

#Συλλέγουμε τις στήλες δεδομένων για κάθε είδος φυτού που μας ενδιαφέρει
x2 , y2 , z2 = xyz("Iris-versicolor")
#Βρίσκουμε το κέντρο συμμετρίας για το σμήνος σημείων του παραπάνω είδους φυτού.
x2c , y2c , z2c = centroid_3d(x2,y2,z2)

#Συλλέγουμε τις στήλες δεδομένων για κάθε είδος φυτού που μας ενδιαφέρει
x3 , y3 , z3 = xyz("Iris-virginica")
#Βρίσκουμε το κέντρο συμμετρίας για το σμήνος σημείων του παραπάνω είδους φυτού.
x3c , y3c , z3c = centroid_3d(x3,y3,z3)



fig = plt.figure()
ax = fig.gca(projection='3d')

#Γραφική αναπαράσταση δεδομένων εκάστοτε είδους φυτού (Τρισδιάστατος χώρος με άξονες: Πλάτος πετάλου(x), Μήκος πετάλου(y), Μήκος σεπάλου(z))
ax.scatter(x1, y1, z1,color = "darkgreen" , label="Iris-setosa")
ax.scatter(x1c, y1c, z1c,color = "blue",marker="D" , label="Setosa Centroid")
ax.scatter(x2, y2, z2,color = "magenta" , label="Iris-versicolor")
ax.scatter(x2c, y2c, z2c,color = "blue",marker="*" , label="Versicolor Centroid")
ax.scatter(x3, y3, z3,color = "orange" , label="Iris-virginica")
ax.scatter(x3c, y3c, z3c,color = "blue",marker="x" , label="Virginica Centroid")

#Σχεδιασμός σφαιρών με κέντρο το κέντρο συμμετρίας του εκάστοτε είδους φυτού και ακτίνα την απόσταση από το σημείο αυτό έως το πιο μακρινό σημείο του εκάστοτε σμήνους.
sphere(x1,y1,z1,x1c,y1c,z1c,"g")
sphere(x2,y2,z2,x2c,y2c,z2c,"m")
sphere(x3,y3,z3,x3c,y3c,z3c,"orange")

#Προσθήκη τίτλων αξόνων
ax.set_xlabel("Petal Width (cm)")
ax.set_ylabel("Petal Length (cm)")
ax.set_zlabel("Sepal Length (cm)")
plt.legend(loc=2)

plt.show()
