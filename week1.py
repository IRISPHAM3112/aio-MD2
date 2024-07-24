#Length of vector 
import numpy as np 
def compute_vector_length(vector):
    len_of_vector = np.sqrt(np.sum(np.array(vector)**2))
    return len_of_vector
v = [1,2,3,4]
print(compute_vector_length(v))
#Dot product
# vector @ vector
def compute_dot_product(vector1, vector2):
    result = np.dot(vector1, vector2)
    return result

u = [1,2,3,4]
v = [5,6,7,8]
print(compute_dot_product(u,v))

# vector @ matrix 
def matrix_multi_vector ( matrix , vector ) :
    result = np.dot(matrix, vector)
    return result 
u = np.array([[2,3,4],[1,2,3],[3,4,5]])
v = [5,6,7]
print(matrix_multi_vector ( u , v ))
# matrix @ matrix 
def matrix_multi_matrix ( matrix1 , matrix2) :
    result = matrix1 @ matrix2
    return result 
u = np.array([[2,4],[1,3],[3,5]])
v = np.array([[2,3,4],[1,2,3]])
print(matrix_multi_matrix ( u , v ))

# matrix inverse with 2x2 matrix 
def inverse_matrix(matrix):
    det_A = (matrix[0][0] * matrix[1][1]) - (matrix[0][1]* matrix[1][0])
    if det_A != 0:
        print("A is invertible")
        inverse_A = (1/det_A) * np.array([[matrix[1][1],-matrix[0][1]],[-matrix[1][0],matrix[0][0]]])
        return inverse_A 
    else: print("A is not invertible")
A = np.array([[2,6],[8,-4]])
print(inverse_matrix(A))

#Eigenvector và eigenvalues:
def compute_eigenvalues_eigen_vectors(matrix):
    eigen_values, eigen_vectors = np.linalg.eigvals(matrix)

    return eigen_values, eigen_vectors
def compute_normalize_vector(matrix):
    length = np.linalg.norm(v)
    normalized_vector = v / length
    return normalized_vector
A = np.array([[0.9,0.2],[0.1,0.8]])
print(compute_eigenvalues_eigen_vectors(A))
print(compute_normalize_vector(A))

#cosine similarity 
def compute_cosine(v1,v2):
    v1 = np.array(v1)
    v2 = np.array(v2)

    upper = v1 @ v2
    lower = np.sqrt(np.sum(v1**2)) * np.sqrt(np.sum(v2**2))
    cosine_similarity = upper/lower
    return cosine_similarity
v1 = [1,2,3,4]
v2 = [1,0,3,0]
print(compute_cosine(v1,v2))
