import numpy as np
import scipy.spatial.distance as dst
from scipy import linalg
from sklearn.metrics import mean_squared_error as mse
import matplotlib.pyplot as plt

films = None


def sorting(s, r):
    for i in range(1, len(s)):
        while i > 0 and s[i] > s[i - 1]:
            s[i], s[i - 1] = s[i - 1], s[i]
            for j in range(len(r)):
                r[j][i], r[j][i - 1] = r[j][i - 1], r[j][i]
            i -= 1
    return s, r


def svd(a: np.matrix):
    if a.shape[0] == a.shape[1]:
        s, u = linalg.eig(a)
        s.sort()
        s = np.delete(np.diagflat(np.append(s[::-1], [0])), (len(s),), axis=0)
        v = u.transpose()
    elif a.shape[0] < a.shape[1]:
        m = a * a.transpose()
        s, u = linalg.eig(m)
        s, u = sorting(s, u)
        s = np.sqrt(s)
        s = np.delete(np.diagflat(np.append(s, [0])), (len(s),), axis=0)
        v = np.linalg.pinv(s) * np.matrix(u).transpose() * a
    else:
        m = a.transpose() * a
        s, v = linalg.eig()
        s.sort()
        s = np.delete(np.diagflat(np.append(s[::-1], [0])), (len(s),), axis=0)
        u = a * np.matrix(v).transpose() * np.linalg.pinv(s)
    return u, s, v


def approximate_svd(u, s, v, percent=0.8):
    su = 0
    for i in range(len(s)):
        su += s[i][i]
    cnt_rows = -1
    cur_sum = 0
    for i in range(len(s)):
        cur_sum += s[i][i]
        cnt_rows = i + 1
        if cur_sum / su > percent:
            break
    s = s[0:cnt_rows, 0:cnt_rows]
    u = u[:, 0:cnt_rows]
    v = v[0:cnt_rows, :]
    return u, s, v


def get_list_neighbours_for_new_user_ratings(user_ratings, u, s, v, n=30):
    u_new = np.matrix(user_ratings) * np.matrix(v.transpose()) * np.matrix(np.linalg.pinv(s))
    similarity_users = list()
    for i in range(len(u)):
        cos = dst.cosine(u_new, u[i])
        similarity_users.append((i, cos))
    similarity_users.sort(key=lambda x: x[1])
    return similarity_users[:n]


def get_most_frequent_recommendations(user_ratings, similarity_users, a, n=3):
    frs = np.zeros((1682,))
    for sim_user in similarity_users:
        for i in range(a[sim_user[0]].size):
            if (user_ratings[i] == 0) and (a[sim_user[0], i] != 0):
                frs[i] += 1
    frequents = list()
    cnt = 0
    for i in range(len(frs)):
        if frs[i] != 0:
            frequents.append((i, frs[i]))
            cnt += 1
    frequents.sort(key=lambda x: x[1])
    return frequents[-n:]


def get_highest_sum_of_ratings_recommendations(user_ratings, similarity_users, a, n=3):
    frs = np.zeros((1682,))
    for sim_user in similarity_users:
        for i in range(a[sim_user[0]].size):
            if (user_ratings[i] == 0) and (a[sim_user[0], i] != 0):
                frs[i] += a[sim_user[0], i]
    frequents = list()
    cnt = 0
    for i in range(len(frs)):
        if frs[i] != 0:
            frequents.append((i, frs[i]))
            cnt += 1
    frequents.sort(key=lambda x: x[1])
    return frequents[-n:]


def load_ratings_of_movie_lens():
    ratings = np.zeros((943, 1682))
    train = open('./341/train', 'r').readlines()
    for line in train:
        user, item, rating = line.split(' ')
        ratings[int(user) - 1, int(item) - 1] = int(rating)
    return ratings


def uv_decomposition(a):
    u, s, v = svd(a)
    u, s, v = approximate_svd(u, s, v)
    return u, np.matrix(s)*np.matrix(v)


def load_films():
    global films
    films = dict()
    items = open('./341/u.item', 'r').readlines()
    for item in items:
        info = item.split('|')
        num, name = info[0], info[1]
        films[int(num)] = name


def get_name_of_num(num):
    global films
    return films[num]


def inc_compute_uv_decomposition(a, eta=0.0002, beta=0.02):
    err_arr = []
    u, v = uv_decomposition(a)
    u = np.matrix(np.ones(u.shape))
    v = np.matrix(np.ones(v.shape))
    counter = 0
    while get_r_mse(a, u*v) > 0.1 and counter < 2000:
        for i in range(u.shape[0]):
            for j in range(v.shape[1]):
                err = a[i, j] - np.dot(u[i], v[:, j])[0, 0]
                u[i] = u[i] + eta * (2 * err * v.transpose()[j] - beta * u[i])
                v[:, j] = v[:, j] + eta * ((2 * err * u[i] - beta * v.transpose()[j]).transpose())
        counter += 1
        err_arr.append(get_r_mse(a, u*v))
    return u, v, err_arr


def get_r_mse(a, b):
    return np.sqrt(mse(a, b))


def check_svd():
    load_films()
    movie_lens = np.matrix(load_ratings_of_movie_lens())
    u, s, v = svd(movie_lens)
    u, s, v = approximate_svd(u, s, v)
    new_user = np.zeros((1682,))
    new_user[49] = 5
    new_user[259] = 4
    new_user[263] = 3
    new_user[287] = 4
    new_user[293] = 5
    new_user[302] = 5
    new_user[353] = 5
    new_user[355] = 3
    new_user[356] = 4
    new_user[360] = 5
    mf_rec = get_most_frequent_recommendations(new_user,
                                            get_list_neighbours_for_new_user_ratings(new_user, u, s, v, n=70),
                                            movie_lens, n=5)
    hsr_rec = get_highest_sum_of_ratings_recommendations(new_user,
                                                     get_list_neighbours_for_new_user_ratings(new_user, u, s, v, n=70),
                                                     movie_lens, n=5)
    print('Most-Frequent recommendations:')
    for r in mf_rec[::-1]:
        print(' ', get_name_of_num(r[0]), r)
    print('Highest of sum ratings recommendations:')
    for r in hsr_rec[::-1]:
        print(' ', get_name_of_num(r[0]), r)


def main():
    # u, v, err_arr = inc_compute_uv_decomposition(np.matrix([[4, 1, 1, 4], [1, 4, 2, 0], [2, 1, 4, 5], [1, 4, 1, 0]]))
    # print(u)
    # print(v)
    # print(u*v)
    err = []
    for j in range(1, 15):
        p, q, err_arr = inc_compute_uv_decomposition(np.matrix([[4, 1, 1, 4], [1, 4, 2, 0], [2, 1, 4, 5], [1, 4, 1, 0]]),
                                                     eta=0.001 / j, beta=0.004)
        # plt.plot(err_arr, linewidth=2)
        # plt.xlabel('Iteration, i', fontsize=20)
        # plt.ylabel('RMSE', fontsize=20)
        # plt.title('Components = ', fontsize=20)
        # plt.show()
        print(j)
        err.append(err_arr[-1])
    plt.plot(err, linewidth=2)
    plt.xlabel('Component size, f', fontsize=20)
    plt.ylabel('RMSE', fontsize=20)
    plt.show()
    # check_svd()

if __name__ == "__main__":
    main()
