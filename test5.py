"""

'table1','table2','table3','table4','table5','table6','table7','table8', 'table9', 'table10'
[0.000,   0.0001,   0.0003, 0.0010,   0.0020, 0.0040,  0.0100,   0.0200,   0.0300,  0.0400]

"""






from __future__ import division
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy import interpolate

class Levels:

    def __init__(self):
        pass

    @staticmethod
    def get_levels(v_n, opal_used, bump):

        if bump == 'gen' and opal_used == 'lmc':

            if v_n == 'grad_c' or v_n == 'grad_w' or v_n == 'grad_w_p' or v_n == 'grad_c_p':
                return [10, 50, 100, 200, 400, 800, 1000, 2000, 3000, 4000, 5000]

            if v_n == 'a_p' or v_n == 'a':
                return [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]

            if v_n == 'vinf':
                # return [1.,1.2,1.4,1.6,1.8,2.0,2.2,2.4,2.6,2.8,3.0]
                return [1.4, 1.8, 2.2, 2.6, 3.0]


        if opal_used.split('/')[-1] == 'gal':

            if v_n == 'r':
                return [0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6]
                             # 1.5, 1.6, 1.7, 1.8, 1.9, 2., 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3.0,4.0
            if v_n == 'm':
                levels = [7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]
            if v_n == 'mdot':
                return [-5.5, -5.25, -5., -4.75, -4.5, -4.25, -4, -3.75, -3.5, -3.25, -3.]
                # levels = [-6.0, -5.9, -5.8, -5.7, -5.6, -5.5, -5.4, -5.3, -5.2, -5.1, -5., -4.9, -4.8, -4.7, -4.6, -4.5]
            if v_n == 'l':
                return [5.0, 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7, 5.8, 5.9, 6.0, 6.1, 6.2]
            if v_n == 'lm':
                return [4.0, 4.05, 4.1, 4.15, 4.2, 4.25, 4.3, 4.35, 4.4, 4.45, 4.50, 4.55]
                          # 4.6, 4.65, 4.7, 4.75, 4.8, 4.85, 4.9, 4.95, 5.0]
            if v_n == 't':
                return [5.15, 5.16, 5.17, 5.18, 5.19, 5.20, 5.21, 5.22, 5.23, 5.24, 5.25, 5.26, 5.27, 5.28, 5.29, 5.30]

            if v_n == 'k':
                # return [0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]  # FOR log Kappa
                return [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2]  # FOR log Kappa
            if v_n == 'rho':
                return [-10, -9.5, -9, -8.5, -8, -7.5, -7, -6.5, -6, -5.5] # , -5, -4.5, -4
            # if v_n_z == 'r':   levels = [0.9, 0.95, 1.0, 1.05, 1.1, 1.15, 1.2, 1.25, 1.3, 1.35, 1.4, 1.45, 1.5, 1.55, 1.6,
            #                            1.65, 1.7, 1.75, 1.8, 1.85, 1.9, 1.95, 2.0, 2.05, 2.10, 2.15, 2.20]
            if v_n == 'tau': #
                return [0, 10, 20, 40, 80]
            # [0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6, 2.8, 3.0]

            if v_n == 'm':
                return [7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]

        if opal_used.split('/')[-1]== '2gal':

            if v_n == 'r':
                return [0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.6, 1.8]
                             # 1.5, 1.6, 1.7, 1.8, 1.9, 2., 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3.0,4.0
            if v_n == 'm':
                levels = [7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]
            if v_n == 'mdot':
                return [-5.5, -5.25, -5., -4.75, -4.5, -4.25, -4, -3.75, -3.5, -3.25, -3.]
                # levels = [-6.0, -5.9, -5.8, -5.7, -5.6, -5.5, -5.4, -5.3, -5.2, -5.1, -5., -4.9, -4.8, -4.7, -4.6, -4.5]
            if v_n == 'l':
                return [5.0, 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7, 5.8, 5.9, 6.0, 6.1, 6.2]
            if v_n == 'lm':
                return [4.0, 4.05, 4.1, 4.15, 4.2, 4.25, 4.3, 4.35, 4.4]
                          # 4.6, 4.65, 4.7, 4.75, 4.8, 4.85, 4.9, 4.95, 5.0]
            if v_n == 't':
                return [5.15, 5.16, 5.17, 5.18, 5.19, 5.20, 5.21, 5.22, 5.23, 5.24, 5.25, 5.26, 5.27, 5.28, 5.29, 5.30]

            if v_n == 'k':
                # return [0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]  # FOR log Kappa
                return [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2]  # FOR log Kappa
            if v_n == 'rho':
                return [-10, -9.5, -9, -8.5, -8, -7.5, -7, -6.5, -6, -5.5] # , -5, -4.5, -4
            # if v_n_z == 'r':   levels = [0.9, 0.95, 1.0, 1.05, 1.1, 1.15, 1.2, 1.25, 1.3, 1.35, 1.4, 1.45, 1.5, 1.55, 1.6,
            #                            1.65, 1.7, 1.75, 1.8, 1.85, 1.9, 1.95, 2.0, 2.05, 2.10, 2.15, 2.20]
            if v_n == 'tau':
                return [0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6, 2.8, 3.0]
            if v_n == 'm':
                return [7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]

        if opal_used.split('/')[-1] == 'lmc' or opal_used.split('/')[-1] == 'smc':

            # if bump == 'HeII':
            #     if v_n == 'mdot':
            #         return [-6.0, -5.75, -5.5, -5.25, -5., -4.75, -4.5]

            if v_n == 'r':
                return [0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2., 2.1]
            if v_n == 'm':
                levels = [7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]
            if v_n == 'mdot':
                return [-6.0, -5.75, -5.5, -5.25, -5., -4.75, -4.5, -4.25, -4, -3.75, -3.5] # , -4.25, -4, -3.75, -3.5
                # levels = [-6.0, -5.9, -5.8, -5.7, -5.6, -5.5, -5.4, -5.3, -5.2, -5.1, -5., -4.9, -4.8, -4.7, -4.6, -4.5]
            if v_n == 'l':
                return [5.0, 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7, 5.8, 5.9, 6.0, 6.1, 6.2, 6.3, 6.4]
            if v_n == 'lm':
                return [4.0, 4.05, 4.1, 4.15, 4.2, 4.25, 4.3, 4.35, 4.4, 4.45,
                          4.5, 4.55, 4.6, 4.65, 4.7, 4.75, 4.8, 4.85, 4.9, 4.95, 5.0]
            if v_n == 't':
                return [5.15, 5.16, 5.17, 5.18, 5.19, 5.20, 5.21, 5.22, 5.23, 5.24, 5.25, 5.26, 5.27, 5.28, 5.29, 5.30]

            if v_n == 'k':
                # return [0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]  # FOR log Kappa
                return [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2]  # FOR log Kappa
            if v_n == 'rho':
                return [-10, -9.5, -9, -8.5, -8, -7.5, -7, -6.5, -6, -5.5, -5, -4.5, -4]
            # if v_n_z == 'r':   levels = [0.9, 0.95, 1.0, 1.05, 1.1, 1.15, 1.2, 1.25, 1.3, 1.35, 1.4, 1.45, 1.5, 1.55, 1.6,
            #                            1.65, 1.7, 1.75, 1.8, 1.85, 1.9, 1.95, 2.0, 2.05, 2.10, 2.15, 2.20]

            if v_n == 't_eff':
                return [4.0, 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 4.8, 4.9, 5.0, 5.1]

            if v_n == 'tau':
                return [2, 4, 8, 16, 32]
                # return [0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6, 2.8, 3.0]

            if bump == 'HeII':
                if v_n == 'a_p' or v_n == 'a':
                    return [0.02, 0.05, 0.1, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.0]

                if v_n == 'grad_c' or v_n == 'grad_w' or v_n == 'grad_w_p' or v_n == 'grad_c_p':
                    return [1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000]

        if v_n == 'r':
            return [0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2., 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3.0,4.0]

        if v_n == 'm':
            return [5, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]

        if v_n == 'lm':
            return [4.0, 4.05, 4.1, 4.15, 4.2, 4.25, 4.3, 4.35, 4.4, 4.45,
                          4.5, 4.55, 4.6, 4.65, 4.7, 4.75, 4.8, 4.85, 4.9, 4.95, 5.0]

        if v_n == 'vrho': return [-4.4, -4.0, -3.6, -3.2, -2.8, -2.4, -2.0, -1.6, -1.2, -0.8, -0.4, 0.4]
            # return [0.4, 0.0, -0.4, -0.8, -1.2, -1.6, -2.0, -2.4, -2.8, -3.2, -3.6, -4.0, -4.4]

        if v_n == 'Ys' or v_n == 'ys':
            return [0.5, 0.55, 0.6, 0.65, 0.7,0.75,0.8,0.85,0.9,0.95,1.0]

        if v_n == 'm_env':
            return [-10.0,-9.9,-9.8,-9.7,-9.6,-9.5,-9.4,-9.3,-9.2,-9.1,-9.0]
        if v_n == 'r_env':
            return [0.0025,0.0050,0.0075,0.01,0.0125,0.0150,0.0175]

        if v_n == 't_eff':
            return [4.0, 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 4.8, 4.9, 5.0, 5.1, 5.2, 5.3, 5.4]

        if v_n == 'r_eff':
            return [1, 2, 3, 4, 5, 6, 7, 8, 9, 10 ,11 ,12, 13, 14 , 16, 18, 20]

        if v_n == 'log_tau':
            return [0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6, 2.8, 3.0]

        if v_n == 'grad_c' or v_n == 'grad_w' or v_n == 'grad_w_p' or v_n == 'grad_c_p':
            return [10, 50, 100, 200, 400, 600, 800, 1000, 1200]

        if v_n == 'L/Ledd':
            return [0.8,0.9,0.95,0.98, 0.99, 1.00, 1.01, 1.02, 1.03, 1.04, 1.05, 1.06, 1.07]

        if v_n == 'vinf':
            # return [1.,1.2,1.4,1.6,1.8,2.0,2.2,2.4,2.6,2.8,3.0]
            return [1.4, 1.8, 2.2, 2.6, 3.0]
            #return [1000,1200,1400,1600,1800,2000,2200,2400,2600,2800,3000]

        if v_n == 'a_p' or v_n == 'a':
            return [0.02, 0.05, 0.1, 0.15, 0.20, 0.25, 0.30]

        raise NameError('Levels are not found for <{}> Opal:{}'.format(v_n, opal_used))


class Labels:
    def __init__(self):
        pass

    @staticmethod
    def lbls(v_n):
        #solar
        if v_n == 'l':
            return r'$\log(L/L_{\odot})$'#(L_{\odot})

        if v_n == 'lgaia':
            return r'$\log(L_{GAIA}/L_{\odot})$'#(L_{\odot})

        if v_n == 'r':
            return r'$R(R_{\odot})$'

        if v_n == 'm' or v_n == 'xm':
            return r'$M(M_{\odot})$'

        #sonic and general
        if v_n == 'v' or v_n == 'u':
            return 'v (km/s)'

        if v_n == 'rho':
            return r'$\log(\rho)$'

        if v_n == 'k' or v_n == 'kappa':
            return r'$\kappa$'

        if v_n == 't':
            return r'log(T/K)'

        if v_n == 'ts':
            return r'$\log(T_{s}/K)$'

        if v_n == 'rs':
            return r'$\log(R_{s}/R_{\odot})$'

        if v_n == 'lm':
            return r'$\log(L/M)$'

        if v_n == 'mdot':
            return r'$\log(\dot{M}$)'

        if v_n == 'Yc' or v_n == 'yc':
            return r'$^{4}$He$_{core}$'

        if v_n == 'He4':
            return r'$^{4}$He$_{surf}$'

        if v_n == 'Ys' or v_n == 'ys':
            return r'$^{4}$He$_{surf}$'

        if v_n == 't_eff' or v_n == 'T_eff':
            return r'$\log($T$_{eff}/K)$'

        if v_n == 't_*' or v_n == 'T_*':
            return r'$\log($T$_{*}$/K$)$'

        if v_n == 'r_eff' or v_n == 'R_eff':
            return r'$\log($R$_{eff}/R_{\odot})$'

        if v_n == 'rho':
            return r'$\log(\rho)$'

        if v_n == 'tau':
            return r'$\tau$'


        if v_n == 'Pr':
            return r'$P_{rad}$'

        if v_n == 'Pg':
            return r'$P_{gas}$'

        if v_n == 'Pg/P_total':
            return r'$P_{gas}/P_{total}$'

        if v_n == 'Pr/P_total':
            return r'$P_{rad}/P_{total}$'


        if v_n == 'mfp':
            return r'$\log(\lambda)$'

        if v_n == 'HP' or v_n == 'Hp':
            return r'$H_{p}$'

        if v_n == 'L/Ledd':
            return r'$L/L_{Edd}$'

        if v_n == 'delta_t':
            return r'$t_i - ts_i$'

        if v_n == 'delta_u':
            return r'$u_i - us_i$'

        if v_n == 'delta_grad_u':
            return r'$\nabla_{r<r_s} - \nabla_{r>r_s}$'

        if v_n == 'rt':
            return r'$\log(R_t/R_{\odot})$'


        if v_n == 'r_infl':
            return r'$R_{inflect} (R_{\odot})$'

        if v_n == 'z':
            return r'$z$'

        if v_n == 'log_tau':
            return r'$\log(\tau)$'

        if v_n == 'vinf' or v_n == 'v_inf':
            return r'$v_{\inf}\cdot 10^3$ (km/s)'

        if v_n == 'a' or v_n == 'a_p':
            return  r'$a$ (km/s$^2$)'

        if v_n == 'beta' or v_n == 'b':
            return r'$\beta$'


        if v_n == 'grad_c' or v_n == 'grad_c_p':
            return r'$\nabla u \cdot 10^5$ $(c^{-1})$'


class Save_Load_tables:
    def __init__(self):
        pass

    @staticmethod
    def save_table(d2arr, metal, bump, name, x_name, y_name, z_name, output_dir ='../data/output/'):

        header = np.zeros(len(d2arr)) # will be first row with limtis and
        # header[0] = x1
        # header[1] = x2
        # header[2] = y1
        # header[3] = y2
        # tbl_name = 't_k_rho'
        # op_and_head = np.vstack((header, d2arr))  # arraching the line with limits to the array

        part = '_' + bump + '_' + metal
        full_name = output_dir + name + '_' + part + '.data'  # dir/t_k_rho_table8.data

        np.savetxt(full_name, d2arr, '%.4f', '  ', '\n',
                   '\nINTERPOLATED OPAL {} TABLE for {} relation'.format(part, name), '',
                   '# {} | {} | {} | {} |'
                   .format('_' + bump + '_' + metal, x_name, y_name, z_name))

        # np.savetxt(full_name, d2arr, '%.4f', '  ', '\n',
        #            '\nINTERPOLATED OPAL {} TABLE for {} relation'.format(part, name), '',
        #            '# {} | {} {} {} | {} {} {} | {} | {} | {}'
        #            .format(opal_used, x_name, x1, x2, y_name, y1, y2, z_name, n_int, n_out))

    @staticmethod
    def load_table(name, x_name, y_name, z_name, metal, bump, dir ='../data/output/'):
        part =  '_' + bump + '_' + metal
        full_name = dir + name + '_' + part + '.data'

        f = open(full_name, 'r').readlines()

        boxes = f[0].split('|')

        # print(boxes)
        # r_table = boxes[0].split()[-1]
        # r_x_name = boxes[1].split()[0]
        # x1 = boxes[1].split()[1]
        # x2 = boxes[1].split()[2]
        # r_y_name = boxes[2].split()[0]
        # y1 = boxes[2].split()[1]
        # y2 = boxes[2].split()[2]
        # r_z_name = boxes[3].split()[0]
        # n1 = boxes[4].split()[-1]
        # n2 = boxes[5].split()[-1]

        r_table  = boxes[0].split()[-1]
        r_x_name = boxes[1].split()[-1]
        r_y_name = boxes[2].split()[-1]
        r_z_name = boxes[3].split()[-1]

        if r_table != '_' + bump + '_' + metal:
            raise NameError('Read OPAL | {} | not the same is opal_used | {} |'.format(r_table,  '_' + bump + '_' + metal))

        if x_name != r_x_name:
            raise NameError('Provided x_name: {} not equal to table x_name: {}'.format(x_name, r_x_name))

        if y_name != r_y_name:
            raise NameError('Provided x_name: {} not equal to table x_name: {}'.format(y_name, r_y_name))

        if z_name != r_z_name:
            raise NameError('Provided x_name: {} not equal to table x_name: {}'.format(z_name, r_z_name))

        # if x1 == 'None':
        #     x1 = None
        # else:
        #     x1 = float(x1)
        #
        # if x2 == 'None:':
        #     x2 = None
        # else:
        #     x2 = float(x2)
        #
        # if y1 == 'None':
        #     y1 = None
        # else:
        #     y1 = float(y1)
        #
        # if y2 == 'None':
        #     y2 = None
        # else:
        #     y2 = float(y2)
        #
        # n1 = int(n1)
        # n2 = int(n2)

        print('\t__OPAL_USED: {}, X is {} | Y is {} | Z is {} '
              .format(r_table, r_x_name, r_y_name, r_z_name))

        print('\t__File | {} | is loaded successfully'.format(full_name))

        # global file_table
        file_table = np.loadtxt(full_name, dtype=float)

        return np.array(file_table, dtype='float64') #[x1, x2, y1, y2, n1, n2]

    @staticmethod
    def save_3d_table(d3array, metal, bump, name, t_name, x_name, y_name, z_name, output_dir ='../data/output/'):
        i = 0

        part =  '_' + bump + '_' + metal
        full_name = output_dir + name + '_' + part  + '.data'  # dir/t_k_rho_table8.data

        # np.savetxt(full_name, d2arr, '%.4f', '  ', '\n',
        #            '\nINTERPOLATED OPAL {} TABLE for {} relation'.format(part, name), '',
        #            '# {} | {} | {} | {} |'
        #            .format(opal_used, x_name, y_name, z_name))

        with open(full_name, 'w') as outfile:
            # I'm writing a header here just for the sake of readability
            # Any line starting with "#" will be ignored by numpy.loadtxt
            # outfile.write('# Array shape: {0}\n'.format(d3array.shape))
            outfile.write(
                '# {} | {} | {} | {} | {} | {} | \n'.format(d3array.shape, t_name, x_name, y_name, z_name,  '_' + bump + '_' + metal))

            # Iterating through a ndimensional array produces slices along
            # the last axis. This is equivalent to data[i,:,:] in this case
            for data_slice in d3array:
                # The formatting string indicates that I'm writing out
                # the values in left-justified columns 7 characters in width
                # with 2 decimal places.
                np.savetxt(outfile, data_slice, '%.4f', '  ', '\n',
                           '\n# {}:{} | {} | {} | {} | {}\n'.format(t_name, data_slice[0,0], x_name, y_name, z_name,  '_' + bump + '_' + metal), '',
                           '')
                # np.savetxt(outfile, data_slice, fmt='%-7.2f')

                # Writing out a break to indicate different slices...
                # outfile.write('# \n')
                i = i + 1

    @staticmethod
    def load_3d_table(metal, bump, name, t_name, x_name, y_name, z_name, output_dir='../data/output/'):
        '''
                            RETURS a 3d table
        :param metal:
        :param name:
        :param t_name:
        :param x_name:
        :param y_name:
        :param z_name:
        :param output_dir:
        :return:
        '''


        part = '_' + bump + '_' + metal
        full_name = output_dir + name + '_' + part + '.data'

        with open(full_name, 'r') as f: # reads only first line (to save time)
            first_line = f.readline()


        first_line = first_line.split('# ')[1] # get rid of '# '
        r_shape = first_line.split(' | ')[0]
        r_t_v_n = first_line.split(' | ')[1]
        r_x_v_n = first_line.split(' | ')[2]
        r_y_v_n = first_line.split(' | ')[3]
        r_z_v_n = first_line.split(' | ')[4]
        r_opal  = first_line.split(' | ')[5]

        # --- Checks for opal (metallicity) and x,y,z,t, var_names.

        if r_opal !=  '_' + bump + '_' + metal:
            raise NameError('Read OPAL <{}> not the same is opal_used <{}>'.format(r_opal,  '_' + bump + '_' + metal))
        if t_name != r_t_v_n:
            raise NameError('Provided x_name: {} not equal to table x_name: {}'.format(t_name, r_t_v_n))
        if x_name != r_x_v_n:
            raise NameError('Provided x_name: {} not equal to table x_name: {}'.format(x_name, r_x_v_n))
        if y_name != r_y_v_n:
            raise NameError('Provided x_name: {} not equal to table x_name: {}'.format(y_name, r_y_v_n))
        if z_name != r_z_v_n:
            raise NameError('Provided x_name: {} not equal to table x_name: {}'.format(z_name, r_z_v_n))

        # --- --- Actual load of a table (as a 2d array) --- --- ---

        d2_table = np.loadtxt(full_name)

        # --- --- Reshaping the 2d arrays into 3d Array --- --- ---

        from ast import literal_eval as make_tuple # to convert str(n, n, n) into a tuple(n, n, n)
        shape = make_tuple(r_shape)

        d3_table = d2_table.reshape(shape)

        print('\t__Table {} is read succesfully. Shape is {}'.format(full_name, d3_table.shape))

        return d3_table

    @staticmethod
    def read_genergic_table(file_name, str_col=None):
        '''
        Reads the the file table, returning the list with names and the table
        structure: First Row must be with '#' in the beginning and then, the var names.
        other Rows - table with the same number of elements as the row of var names
        :return:
        :str_col: Name of the volumn with string values
        '''
        table = []
        with open(file_name, 'r') as f:
            for line in f:
                if '#' not in line.split() and line.strip():  # if line is not empty and does not contain '#'
                    table.append(line)

        names = table[0].split()[:]  # getting rid of '#' which is the first element in a row
        num_colls = len(table) - 1  # as first row is of var names

        if len(names) != len(table[1].split()):
            print('\t___Error. Number of vars in list({}) != number of cols in observ.data file({}) '
                  '|Read_Observables, __init__|'.format(len(names), len(table[1].split())))
        print('\t__Note: Data include following paramters:\n\t | {} |'.format(names))

        table.remove(table[0])  # removing the var_names line from the array. (only actual values left)

        if str_col==None:
            tmp = np.zeros(len(names))
            for row in table:
                tmp = np.vstack((tmp, np.array(row.split(), dtype=np.float)))
            table = np.delete(tmp, 0, 0)

            return names, table

        else:
            # In case there is a column with strings in a table.
            # Remove it from the table. Vstack the rest of the table. Remove it name from the 'names'
            # Return all of it
            tmp = np.zeros(len(names) - 1)
            str_col_val = []
            for row in table:
                raw_row = row.split()
                str_col_val.append(raw_row[names.index(str_col)])
                raw_row.remove(raw_row[names.index(str_col)])
                tmp = np.vstack((tmp, np.array(raw_row, dtype=np.float)))

            table = np.delete(tmp, 0, 0)
            names.remove(str_col)

            return names, table, str_col_val


class Constants:

    light_v = np.float( 2.99792458 * (10 ** 10) )      # cm/s
    solar_m = np.float ( 1.99 * (10 ** 33)  )          # g
    solar_l = np.float ( 3.9 * (10 ** 33)  )           # erg s^-1
    solar_r = np.float ( 6.96 * (10 ** 10) )           #cm
    grav_const = np.float ( 6.67259 * (10 ** (-8) )  ) # cm3 g^-1 s^-2
    k_b     =  np.float ( 1.380658 * (10 ** (-16) ) )  # erg k^-1
    m_H     =  np.float ( 1.6733 * (10 ** (-24) ) )    # g
    m_He    =  np.float ( 6.6464764 * (10 ** (-24) ) ) # g
    c_k_edd =  np.float ( 4 * light_v * np.pi * grav_const * ( solar_m / solar_l ) )# k = c_k_edd*(M/L) (if M and L in solar units)

    yr      = np.float( 31557600. )
    smperyear = np.float(solar_m / yr)

    steph_boltz = np.float(5.6704*10**(-5))

    def __init__(self):
        pass


class Physics:
    def __init__(self):
        pass

    @staticmethod
    def get_rho(r_arr, t_arr):

        cols = len(r_arr)  # 28
        raws = len(t_arr)  # 76

        rho = np.zeros((raws, cols))

        for i in range(raws):
            for j in range(cols):
                rho[i, j] = r_arr[j] + 3 * t_arr[i] - 18

        return rho

    @staticmethod
    def loglm_logk(loglm, array = False):
        '''
        For log(l/m) = 4.141 -> log(k) = -0.026
        :param loglm:
        :return:
        '''
        if array:
            res = np.zeros(len(loglm))
            for i in range(len(loglm)):
                res[i] = np.log10(Constants.c_k_edd) + np.log10(1 / (10 ** loglm[i]))
            return res
        else:
            return np.log10(Constants.c_k_edd) + np.log10(1 / (10 ** loglm))

    @staticmethod
    def logk_loglm(logk, dimensions = 0, coeff = 1.0):
        '''
        k_opal = coeff * k_edd; k_edd = 4*pi*c*G*M / L
        For logk = -0.026 -> log(l/m) = 4.141
        :param logk:
        :return:
        '''
        if dimensions == 1:
            res = np.zeros(len(logk))
            for i in range(len(logk)):
                res[i] = np.log10(1 / (10 ** logk[i])) + np.log10(coeff * Constants.c_k_edd)
            return res
        if dimensions == 0:
            return np.log10(1 / (10 ** logk)) + np.log10(coeff * Constants.c_k_edd)

        if dimensions == 2:
            res = np.zeros(( len(logk[:,0]), len(logk[0,:] )))
            for i in range(len(logk[:,0])):
                for j in range(len(logk[0,:])):
                    res[i, j] = np.log10(1 / (10 ** logk[i,j])) + np.log10(coeff * Constants.c_k_edd)
            return res

        else:
            raise ValueError('\t__Error. Wrong number of dimensions. Use 0,1,2. Given: {}. | logk_loglm |'.format(dimensions))

        # --- --- --- LANGER --- --- ---

    @staticmethod
    def get_k1_k2_from_llm1_llm2(t1, t2, l1, l2):
        lm1 = None
        if l1 != None:
            lm1 = Physics.l_to_lm_langer(l1)
        lm2 = None
        if l2 != None:
            lm2 = Physics.l_to_lm_langer(l2)

        if lm1 != None:
            k2 = Physics.loglm_logk(lm1)
        else:
            k2 = None
        if lm2 != None:
            k1 = Physics.loglm_logk(lm2)
        else:
            k1 = None

        print('\t__Provided LM limits ({}, {}), translated to L limits: ({}, {})'.format(lm1, lm2, l1, l2))
        print('\t__Provided T limits ({},{}), and kappa limits ({}, {})'.format(t1, t2, k1, k2))
        return [k1, k2]

    @staticmethod
    def lm_to_l_langer(log_lm):
        '''
        From Langer 1987 paper Mass Lum relation for WNE stars
        :param log_lm:
        :return:
        '''
        a1 = 2.357485
        b1 = 3.407930
        c1 = -0.654431
        a2 = -0.158206
        b2 = -0.053868
        c2 = 0.055467
        # f1 = a1 + b1*lm + c1*(lm**2)
        # f2 = a2 + b2*ll + c2*(ll**2)

        d = log_lm + a2
        print((1 - b2))
        disc = ((b2 - 1) ** 2 - 4 * c2 * d)
        #
        res = ((- (b2 - 1) - np.sqrt(disc)) / (2 * c2))

        return res

    @staticmethod
    def l_to_lm_langer(log_l):
        '''
        From Langer 1987 paper Mass Lum relation for WNE stars
        :param log_l:
        :return: log_lm
        '''
        a1 = 2.357485
        b1 = 3.407930
        c1 = -0.654431
        a2 = -0.158206
        b2 = -0.053868
        c2 = 0.055467
        return (-a2 -(b2 -1)*log_l - c2*(log_l**2) )

    @staticmethod
    def sound_speed(t, mu, array = False):
        '''

        :param t_arr: log(t) array
        :param mu: mean molecular weight, by default 1.34 for pure He ionised matter
        :return: array of c_s (sonic vel) in cgs

        Test: print(Physics.sound_speed(5.2) / 100000) should be around 31
        '''

        if array:
            if len(mu)!= len(t):
                raise ValueError('\t__Error. Mu and t must be arrays of the same size: (mu: {}; t: {})'.format(mu, t) )
            res = np.zeros(len(t))
            for i in range(len(t)):
                res[i] = (np.sqrt(Constants.k_b*(10**t[i]) / (mu[i] * Constants.m_H))) / 100000
            return res
        else:
            return (np.sqrt(Constants.k_b*(10**t) / (mu * Constants.m_H))) / 100000# t is given in log


    # --- --- --- --- --- --- --- MDOT --- --- --- --- --- --- ---
    @staticmethod
    def vrho_formula(t, rho, mu):
        # assuming that mu is a constant!

        # c = np.log10(4 * 3.14 * ((Constants.solar_r) ** 2) / Constants.smperyear)
        # c2 = c + np.log10(r_s ** 2)
        c2 = 0
        return (rho + c2 + np.log10(Physics.sound_speed(t, mu, False) * 100000))

    @staticmethod
    def get_vrho(t, rho, dimensions=1, mu=np.array([1.34])):
        '''
        :param t:
        :param rho:
        :param dimensions:
        :param r_s:
        :param mu:
        :return:             THIS FUNCTION

                     |           rho*v                          |           Mdot
               L/M   |                                     L/M  |
             .*      |                              ->          |
        kappa        |                              ->      or  |
             *-.     |                                          |
                L    |                                      L   |
                     |____________________________              |________________________
                                                ts                                      ts
        '''

        if int(dimensions) == 0:
            return Physics.vrho_formula(t, rho, mu)

        if int(dimensions) == 1:
            res = np.zeros(len(t))
            for i in range(len(t)):
                res[i] = Physics.vrho_formula(t[i], rho[i], mu)  # pissibility to add mu[i] if needed
            return res

        if int(dimensions) == 2:

            cols = len(rho[0, :])
            rows = len(rho[:, 0])
            m_dot = np.zeros((rows, cols))

            for i in range(rows):
                for j in range(cols):
                    m_dot[i, j] = Physics.vrho_formula(t[j], rho[i, j],
                                                       mu)  # + c + np.log10(Physics.sound_speed(t[j], mu, False)*100000))
            return m_dot
        else:
            raise ValueError(' Wrong number of dimensions. Use 0,1,2. Given: {} '.format(dimensions))

    @staticmethod
    def vrho_mdot(vrho, r_s, r_s_for_t_l_vrho):
        # if vrho is a float and r_s is a float - r_s_for_t_l_vrho = ''
        # if vrho is 1darray and r_s is a float - r_s_for_t_l_vrho = ''
        # if vrho is 2darray and r_s is a float - r_s_for_t_l_vrho = ''

        # if vrho is 1darray and r_s is a 1d array - r_s_for_t_l_vrho = '-'

        # if vrho is 2darray and r_s is a 1d array - r_s_for_t_l_vrho = 't' or 'l' to change columns or rows of vrho

        # if vrho is 2darray and r_s is a 2d array - r_s_for_t_l_vrho = 'tl' to change columns and rows of vrho

        # r_s_for_t_l_vrho = '', 't', 'l', 'lm', 'vrho'

        # -------------------- --------------------- ----------------------------
        c = np.log10(4 * 3.14 * ((Constants.solar_r) ** 2) / Constants.smperyear)

        if r_s_for_t_l_vrho == '':  # vrho is a constant
            mdot = None
        else:
            if r_s_for_t_l_vrho == '-':  # vrho is a 1d array
                mdot = np.zeros(vrho.shape)
            else:
                mdot = np.zeros((vrho.shape))  # vrho is a 2d array

        if r_s_for_t_l_vrho == '':  # ------------------------REQUIRED r_s = float
            c2 = c + np.log10(r_s ** 2)
            mdot = vrho + c2

        if r_s_for_t_l_vrho == '-':
            if len(r_s) != len(vrho): raise ValueError('len(r_s)={}!=len(vrho)={}'.format(len(r_s), len(vrho)))
            for i in range(len(vrho)):
                mdot[i] = vrho[i] + c + np.log10(r_s[i] ** 2)

        if r_s_for_t_l_vrho == 'l' or r_s_for_t_l_vrho == 'lm':  # ---r_s = 1darray
            if len(r_s) != len(vrho[:, 0]): raise ValueError(
                'len(r_s)={}!=len(vrho[:, 0])={}'.format(len(r_s), len(vrho[:, 0])))
            for i in range(len(vrho[:, 0])):
                mdot[i, :] = vrho[i, :] + c + np.log10(r_s[i] ** 2)

        if r_s_for_t_l_vrho == 't' or r_s_for_t_l_vrho == 'ts':  # ---r_s = 1darray
            if len(r_s) != len(vrho[0, :]): raise ValueError(
                'len(r_s)={}!=len(vrho[0, :])={}'.format(len(r_s), len(vrho[0, :])))
            for i in range(len(vrho[0, :])):
                mdot[:, i] = vrho[:, i] + c + np.log10(r_s[i] ** 2)

        if r_s_for_t_l_vrho == 'tl':  # ---------------------REQUIRED r_s = 2darray
            if r_s.shape != vrho.shape: raise ValueError('r_s.shape {} != vrho.shape {}'.format(r_s.shape, vrho.shape))
            cols = len(vrho[0, :])
            rows = len(vrho[:, 0])
            mdot = np.zeros((rows, cols))

            for i in range(rows):
                for j in range(cols):
                    mdot[i, j] = vrho[i, j] + c + np.log10(r_s[i, j] ** 2)

        return mdot

    @staticmethod
    def rho_mdot(t, rho, dimensions=1, r_s=1., mu=1.34):
        '''
        NOTE! Rho2d should be .T as in all outouts it is not .T in Table Analyze
        :param t: log10(t[:])
        :param rho: log10(rho[:,:])
        :param r_s:
        :param mu:
        :return:
        '''

        # c = np.log10(4*3.14*((r_s * Constants.solar_r)**2) / Constants.smperyear)
        c = np.log10(4 * 3.14 * ((Constants.solar_r) ** 2) / Constants.smperyear) + np.log10(r_s ** 2)

        if int(dimensions) == 0:
            return (rho + c + np.log10(Physics.sound_speed(t, mu, False) * 100000))

        if int(dimensions) == 1:
            m_dot = np.zeros(len(t))
            for i in range(len(t)):
                m_dot[i] = (rho[i] + c + np.log10(Physics.sound_speed(t[i], mu, False) * 100000))
            return m_dot

        if int(dimensions) == 2:
            cols = len(rho[0, :])
            rows = len(rho[:, 0])
            m_dot = np.zeros((rows, cols))

            for i in range(rows):
                for j in range(cols):
                    m_dot[i, j] = (rho[i, j] + c + np.log10(Physics.sound_speed(t[j], mu, False) * 100000))
            return m_dot
        else:
            raise ValueError('\t__Error. Wrong number of dimensions. Use 0,1,2. Given: {} | m_dot |'.format(dimensions))

    @staticmethod
    def mdot_rho(t, mdot, dimensions=1, r_s=1., mu=1.34):
        smperyear = Constants.solar_m / Constants.yr

        c = np.log10(4 * 3.14 * ((r_s * Constants.solar_r) ** 2) / smperyear)

        if int(dimensions) == 0:
            return (mdot - c - np.log10(Physics.sound_speed(t, mu, False) * 100000))

        if int(dimensions) == 1:
            m_dot = np.zeros(len(t))
            for i in range(len(t)):
                m_dot[i] = (mdot[i] - c - np.log10(Physics.sound_speed(t[i], mu, False) * 100000))
            return m_dot

        if int(dimensions) == 2:
            cols = len(mdot[0, :])
            rows = len(mdot[:, 0])
            m_dot = np.zeros((rows, cols))

            for i in range(rows):
                for j in range(cols):
                    m_dot[i, j] = (mdot[i, j] - c - np.log10(Physics.sound_speed(t[j], mu, False) * 100000))
            return m_dot
        else:
            raise ValueError('\t__Error. Wrong number of dimensions. Use 0,1,2. Given: {} | mdot_rho |'.format(dimensions))
    # --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---


class Math:
    def __init__(self):
        pass

    @staticmethod
    def x_y_z_sort(x_arr, y_arr, z_arr=np.empty(0,), sort_by_012=0):
        '''
        RETURNS x_arr, y_arr, (z_arr) sorted as a matrix by a row, given 'sort_by_012'
        :param x_arr:
        :param y_arr:
        :param z_arr:
        :param sort_by_012:
        :return:
        '''

        if not z_arr.any() and sort_by_012 < 2:
            if len(x_arr) != len(y_arr):
                raise ValueError('len(x)[{}]!= len(y)[{}]'.format(len(x_arr), len(y_arr)))

            x_y_arr = []
            for i in range(len(x_arr)):
                x_y_arr = np.append(x_y_arr, [x_arr[i], y_arr[i]])

            x_y_sort = np.sort(x_y_arr.view('float64, float64'), order=['f{}'.format(sort_by_012)], axis=0).view(
                np.float)
            x_y_arr_shaped = np.reshape(x_y_sort, (int(len(x_y_sort) / 2), 2))
            return x_y_arr_shaped[:, 0], x_y_arr_shaped[:, 1]

        if z_arr.any():
            if len(x_arr) != len(y_arr) or len(x_arr) != len(z_arr):
                raise ValueError('len(x)[{}]!= len(y)[{}]!=len(z_arr)[{}]'.format(len(x_arr), len(y_arr), len(z_arr)))

            x_y_z_arr = []
            for i in range(len(x_arr)):
                x_y_z_arr = np.append(x_y_z_arr, [x_arr[i], y_arr[i], z_arr[i]])

            x_y_z_sort = np.sort(x_y_z_arr.view('float64, float64, float64'), order=['f{}'.format(sort_by_012)],
                                 axis=0).view(
                np.float)
            x_y_z_arr_shaped = np.reshape(x_y_z_sort, (int(len(x_y_z_sort) / 3), 3))
            return x_y_z_arr_shaped[:, 0], x_y_z_arr_shaped[:, 1], x_y_z_arr_shaped[:, 2]

    @staticmethod
    def solv_inter_row(arr_x, arr_y, val):
        '''
        FUNNY! but if val == arr_y[i] exactly, the f.roots() return no solution for some reaon :)
        :param arr_x:
        :param arr_y:
        :param val:
        :return:
        '''

        arr_x, arr_y = Math.x_y_z_sort(arr_x, arr_y, np.empty(0,))

        if arr_x.shape != arr_y.shape:
            print("y_arr:({} to {}), x_arr: ({} to {}) find: y_val {} ."
                  .format("%.2f"%arr_y[0],"%.2f"%arr_y[-1], "%.2f"%arr_x[0], "%.2f"%arr_x[-1],"%.2f"%val))
            raise ValueError

        if val in arr_y:
            return np.array(arr_x[np.where(arr_y == val)])
        else:
            # Errors.is_a_bigger_b(val,arr_y[-1],"|solv_inter_row|", True, "y_arr:({} to {}), can't find find: y_val {} .".format("%.2f"%arr_y[0],"%.2f"%arr_y[-1],"%.2f"%val))
            red_arr_y = arr_y - val

            # new_x = np.mgrid[arr_x[0]:arr_x[-1]:1000j]
            # f1 = interpolate.UnivariateSpline(arr_x, red_arr_y, s=0)
            # new_y = f1(new_x)

            # f = interpolate.InterpolatedUnivariateSpline(new_x, new_y)

            # new_y = f1(new_x)
            # f = interpolate.UnivariateSpline(new_x, new_y, s=0)
            # print("y_arr:({} to {}), x_arr: ({} to {}) find: y_val {} ."
            #       .format("%.2f"%arr_y[0],"%.2f"%arr_y[-1], "%.2f"%arr_x[0], "%.2f"%arr_x[-1],"%.2f"%val))

            red_arr_y = np.array(red_arr_y, dtype=np.float)
            arr_x = np.array(arr_x, dtype=np.float)



            f = interpolate.InterpolatedUnivariateSpline(arr_x, red_arr_y)
            # print("y_arr:({} to {}), can't find find: y_val {} .".format("%.2f"%arr_y[0],"%.2f"%arr_y[-1],"%.2f"%val))
            # f = interpolate.UnivariateSpline(arr_x, red_arr_y, s = 0)
            return f.roots() # x must be ascending to get roots()!

    @staticmethod
    def find_nearest_index(array, value):
        ''' Finds index of the value in the array that is the closest to the provided one '''
        idx = (np.abs(array - value)).argmin()
        return idx

    @staticmethod
    def combine(x, y, xy, corner_val = None):
        '''creates a 2d array  1st raw    [0, 1:]-- x -- density      (log)
                               1st column [1:, 0] -- y -- lemperature (log)
                               Matrix     [1:,1:] -- xy --Opacity     (log)
           0th element in 1st raw (column) - can be used a corner value

        '''
        x = np.array(x)
        y = np.array(y)
        xy = np.array((xy))

        if len(x) != len(y):
            print('\t__Warning. x({}) != y({}) (combine)'.format(len(x), len(y)))
        if len(x) != len(xy[0, :]):
            raise ValueError('\t__Warning. x({}) != xy[0, :]({}) (combine)'.format(len(x), len(xy[0, :])))
        if len(y) != len(xy[:, 0]):
            raise ValueError('\t__Warning. y({}) != xy[:, 0]({}) (combine)'.format(len(y), len(xy[:, 0])))

        res = np.insert(xy, 0, x, axis=0)
        new_y = np.insert(y, 0, 0, axis=0)  # inserting a 0 to a first column of a
        res = np.insert(res, 0, new_y, axis=1)

        if corner_val != None:
            res[0, 0] = corner_val

        return res

    @staticmethod
    def interp_row(x_arr, y_arr, new_x_arr):
        '''
            Uses 1d spline interpolation to give set of values new_y for provided
            cooednates x and y and new coordinates x_new (s - to be 0)
        '''
        # print(x_arr)
        f = interpolate.InterpolatedUnivariateSpline(x_arr, y_arr)
        # f = interpolate.interp1d(x_arr, y_arr, kind='cubic')


        return f(new_x_arr)

    @staticmethod
    def invet_to_ascending_xy(d2array):
        x = np.array(d2array[0, 1:])
        y = np.array(d2array[1:, 0])
        z = np.array(d2array[1:, 1:])

        if x[0] > x[-1]:
            print('\t__Note: Inverting along X axis')
            x = x[::-1]
            z = z.T
            z = z[::1]
            z = z.T

        if y[0] > y[-1]:
            print('\t__Note: Inverting along Y axis')
            y = y[::-1]
            z = z[::-1]

        print(x.shape, y.shape, z.shape)
        return Math.combine(x, y, z)


class Read_Table:
    '''
        This class reads the 2D OPAL table, where
        0th raw is values of R
        0th column are temperatures
        else - values of opacity
        Everything in log10()
    '''

    def __init__(self, table_name):
        '''
        :param table_name: example ./opal/table1 extensiton is .data by default
        '''

        self.table_name = table_name

        f = open(fname, 'r').readlines()
        len1d = f.__len__()
        len2d = f[0].split().__len__()
        table = np.zeros((len1d, len2d))
        for i in range(len1d):
            table[i, :] = np.array(f[i].split(), dtype=float)

        r = table[0, 1:]
        t = table[1:, 0]
        rho = Physics.get_rho(r, t)
        k = table[1:, 1:]
        #
        print(r.shape)
        print(t.shape)

        print(rho.shape)
        print(k.shape)

        # ---
        self.r = r
        self.t = t
        self.kappas = k
        self.rho = rho
        self.table = table


class Row_Analyze:

    mask = 9.999 # mask vale can be adjusted!
    crit_for_smooth = 2 # by  what factor two consequent values can be diff
                        # to still be smooth

    def __init__(self):
        pass

    #---------------------STEP_1-----------------------
    # Cut the 'mask' values
    @staticmethod
    def cut_mask_val(x_row, y_row):
        '''
            Removes mask values from opal_raw array and
            corresponding elements from rho_raw array
        '''

        x_row = np.array(x_row)
        y_row = np.array(y_row)

        arr_mask = []

        for i in range(len(y_row)):  # might be a problem with not enough elements. Put -1
            if y_row[i] == Row_Analyze.mask: # take val from class
                arr_mask = np.append(arr_mask, i)  # elements to be deleted from an array

        if any(arr_mask):
            print('\t___Note: Mask Values = ',Row_Analyze.mask,' removed at: ', arr_mask)

        y_row = np.delete(y_row, arr_mask)  # removing mask elements
        x_row = np.delete(x_row, arr_mask)

        return np.vstack((x_row, y_row))

    #---------------------STEP_2-----------------------
    # Cut the largest smooth area (never actually tested!!!)
    # Inapropriate way of doing it anyway...
    @staticmethod
    def get_smooth(x_row, y_row):

        # 2. Check for not smoothness
        arr_excess = []
        arr_excess = np.append(arr_excess, 0)
        delta = np.diff(y_row)
        delta_cr = Row_Analyze.crit_for_smooth * np.sum(delta) / (len(delta))  # mean value of all elements

        for i in range(len(delta)):  # delta -2 as delta has 1 less element than opal
            if (delta[i] == delta_cr):
                arr_excess = np.append(arr_excess, i)

        arr_excess = np.append(arr_excess, (len(y_row)))

        # 3. Selecting the biggest smooth region: (not checked!)
        if (len(arr_excess) > 2):
            print('\t___Warning! Values are no smooth. Taking the largest smooth area, | get_smooth |')
            diff2 = np.diff(arr_excess)
            ind_max = np.argmax(diff2)
            ind_begin = arr_excess[ind_max] + 1  # start of the biggest smooth region
            ind_end = arr_excess[ind_max + 1]  # end of the biggest smooth region

            if (ind_begin > (len(y_row) - 1)):   raise ValueError('\t___Error in fingin the start of the smooth area. | get_smooth |')
            if (ind_end > (len(y_row) - 1)):   raise ValueError('\t___Error in fingin the end of the smooth area. | get_smooth |')
            print('\t___Warning! only elements:', ind_begin, '->', ind_end, '(', ind_begin - ind_end, 'out of ',
                  len(y_row), 'are taken, |get_smooth|)')

            # print(ind_begin, ind_end)
            y_row = y_row[int(ind_begin):int(ind_end)]  # selecting the smooth region
            x_row = x_row[int(ind_begin):int(ind_end)]


        return np.vstack((x_row, y_row))

    # ---------------------STEP_3-----------------------
    # Cut the repeating values in the beginning of the raw
    # repetitions in the middel are not addressed!
    @staticmethod
    def cut_rep_val(x_row, y_row):
        '''
            Warning! Use with caution.
            Repetition in the middle of the data is not treated.
        :param x_row:
        :param y_row:
        :return:
        '''
        # 4. Selecting and removing the repeating regions (same value)
        delta = np.diff(y_row)  # redefine as length of an array has changed
        arr_zero = []
        if (delta[0] == 0):
            for i in range(len(delta)):
                if (delta[i] == 0):
                    arr_zero = np.append(arr_zero, i)  # collecting how many elements in the beginning are the same
                else:
                    break

        if (len(arr_zero) != 0): print('\t___Warning! Repetitions', arr_zero, ' in the beginning -> removed | cut_rep_val |')

        y_row = np.delete(y_row, arr_zero)  # removing the repetition in the beginning
        x_row = np.delete(x_row, arr_zero)

        # checking if there is repetition inside
        delta = np.diff(y_row)  # again - redefenition
        arr_zero2 = []
        for i in range(len(delta)):
            if (delta[i] == 0):
                arr_zero2 = np.append(arr_zero2, i)

        if (len(arr_zero2) > 0): print('\t___Warning! repeating values: ', arr_zero2,
                                       'inside an array -> NOT REMOVED!| cut_rep_val |')

        return np.vstack((x_row, y_row))

    # All abve methods together, performed if conditions are True
    @staticmethod
    def clear_row(x_row, y_row, cut_mask=True, cut_rep=True, cut_smooth=True):

        x_tmp = x_row
        y_tmp = y_row

        if cut_mask:
            no_masks = Row_Analyze.cut_mask_val(x_tmp, y_tmp)
            x_tmp = no_masks[0, :]
            y_tmp = no_masks[1, :]  # it will update them, if this option is chosen

        if cut_smooth:
            smooth = Row_Analyze.get_smooth(x_tmp, y_tmp)
            x_tmp = smooth[0, :]
            y_tmp = smooth[1, :]  # it will update them, if this option is chosen

        if cut_rep:
            no_rep = Row_Analyze.cut_rep_val(x_tmp, y_tmp)
            x_tmp = no_rep[0, :]
            y_tmp = no_rep[1, :]  # it will update them, if this option is chosen

        return np.vstack((x_tmp, y_tmp))

    # ---------------------STEP_4-----------------------
    # Identefy the cases and solve for the case
    # repetitions in the middel are not addressed!
    @staticmethod
    def get_case_limits(x_row, y_row, n_anal):
        '''
        :param x_row:
        :param y_row:
        :param n_anal: Here to be 1000 as it is only searching for limits. More points - more precise y1,y2
        :return: array[case, lim_op1, lim_y2]
        '''
        case = -1  # -1 stands for unknown
        x_row = np.array(x_row)
        y_row = np.array(y_row)
        y1 = y_row.min()
        y2 = y_row.max()
        lim_y1 = 0  # new lemeits from interploation
        lim_y2 = 0

        # print('\t Opal region: ', y1, ' ->', y2)

        new_y_grid = np.mgrid[y1:y2:(n_anal) * 1j]

        singl_i = []
        singl_sol = []
        singl_y = []
        db_i = []
        db_y = []
        db_sol_1 = []
        db_sol_2 = []
        exc_y_occur = []

        for i in range(1, n_anal - 1):  # Must be +1 -> -1 or there is no solution for the last and first point(

            sol = Math.solv_inter_row(x_row, y_row, new_y_grid[i])

            if (len(sol) == 0):
                print('\t___ERROR! At step {}/{} No solutions found | get_case_limits | \n '
                      ' k_row:({}, {}), k_grid_point: ({})'.format(i, n_anal - 1, y_row[0], y_row[-1], new_y_grid[i]))

                # sys.exit('\t___Error: No solutions Found in | Row_Analyze, get_case_limits |')

            if (len(sol) == 1):
                singl_i = np.append(singl_i, i)  # list of indexis of grid elements
                singl_sol = np.append(singl_sol, sol)  # list of kappa values
                singl_y = np.append(singl_y, new_y_grid[i])

            if (len(sol) == 2):
                db_i = np.append(db_i, int(i), )
                db_sol_1 = np.append(db_sol_1, sol[0])
                db_sol_2 = np.append(db_sol_2, sol[1])
                db_y = np.append(db_y, new_y_grid[i])

            if (len(sol) > 2):  # WARNING ! I Removed the stop signal
                exc_y_occur = np.append(exc_y_occur, new_y_grid[i])
                print('\t___Warning! At step', i, 'More than 2 solutions found | get_case_limits |', sol)
        #            sys.exit('___Error: more than 2 solution found for a given kappa.')



        print('\t__Note: single solutions for:', len(singl_y), ' out of ', n_anal - 2, ' elements | get_case_limits |')
        print('\t__Note: Double solutions for:', len(db_y), ' out of ', n_anal - 2 - len(singl_y), ' expected')

        # Check if there are several regions of degeneracy:
        delta = np.diff(db_i)
        for i in range(len(delta)):
            if (delta[i] > 1):
                raise ValueError('\t___Error! Found more than 1 degenerate region. Treatment is not prescribed | get_case_limits |')

        # Defining the cases - M, A, B, C, D and limits of the opal in each case.
        if (len(db_i) == 0 and len(exc_y_occur) == 0):
            case = 0  # Monotonic CASE DETERMINED
            lim_y1 = singl_y[0]
            lim_y2 = singl_y[-1]
            print('\n\t<<<<< Case 0 (/) >>>>>\n')

        # If there is a degenerate region - determin what case it is:
        # and Remove the part of the kappa (initial) that we don't need:
        if (len(db_i) != 0):
            mid = db_sol_1[len(db_sol_1) - 1] + (db_sol_2[len(db_sol_2) - 1] - db_sol_1[len(db_sol_1) - 1]) / 2

            if ((db_sol_2[-1] - db_sol_1[-1]) > (db_sol_2[0] - db_sol_1[0])):
                if (singl_sol[len(singl_sol) - 1] > mid):
                    case = 1
                    lim_y1 = np.array([singl_y[0], db_y[0]]).min()
                    lim_y2 = singl_y[-1]

                    print('\n\t<<<<< Case 1 (-./) >>>>>\n')
                else:
                    case = 2
                    # print(db_i)
                    lim_y1 = np.array([new_y_grid[int(db_i[0])], singl_y[0]]).min()
                    lim_y2 = new_y_grid[int(db_i.max())]

                    print('\n\t <<<<< Case 2 (-*\) >>>>>\n')
                    print('\t___Warning! Case (-*\) reduces the k region to: ', "%.2f" % lim_y1, ', ', "%.2f" % lim_y2)
            else:
                if (singl_sol[0] > mid):
                    case = 3

                    lim_y1 = new_y_grid[int(db_i.min())]
                    lim_y2 = np.array([new_y_grid[int(db_i.max())], singl_y[len(singl_y) - 1]]).max()

                    print('\n\t <<<<< Case 3 (\.-) >>>>>\n')
                    print('\t___Warning! Case (\.-) limits the range of kappa to:', "%.2f" % lim_y1, ', ', "%.2f" % lim_y2)
                else:
                    lim_y1 = np.array(singl_y[0], db_y[0]).min()
                    lim_y2 = np.array([singl_y[-1], db_y[-1]]).max()
                    case = 4

                    print('\n\t <<<<< Case 4 (/*-) >>>>>\n')

        if (len(db_i) == 0 and len(exc_y_occur) != 0):
            lim_y1 = singl_y[0]
            lim_y2 = singl_y[-1]
            case = 5
            print('\n\t<<<<< Warning! Case unidentified! (Case 5) >>>>>\n')

        # db_sol = np.hstack(([db_i[0], db_i[-1]], [db_sol_1[0], db_sol_1[-1]], [db_sol_2[0], db_sol_2[-1]], [db_y[0], db_y[-1]]))
        # sl_sol = np.hstack(([singl_i[0], singl_i[-1]], [singl_sol[0], singl_sol[-1]], [singl_y[0], singl_y[-1]]))
        # exc = np.array([exc_y_occur[0], exc_y_occur[-1]])
        # db_sol =[db_i: (1st, last), db_sol_1: (1st, last), db_sol_2: (1st, last), db_y: (1st last)]
        # sl_sol = [singl_i: (1st last), sing_sol: (first, last), sing_op: (first, last)]
        # exc_global = [1st elemnt, last element]

        return np.array([case, lim_y1, lim_y2])

    # ---------------------STEP_5-----------------------
    # Solve for a given case
    # Warning! Case 5 is unidentified
    @staticmethod
    def case_0(x_row, y_row, y_grid, depth):
        f_opal = []
        f_rho = []
        for i in range(len(y_grid)):

            # kap = kappa_grid[i]+0.0000000000000000000000000000000000000000000000000000000000000000000000000000000000001

            # f = interpolate.UnivariateSpline(rho_row, np.array(kappa_row - kap), s=0)
            sol = Math.solv_inter_row(x_row, y_row, y_grid[i])
            # sol = f.roots()
            if len(sol) == 0:
                print('\t__Error. No solutions in |case_0|. kappa_row - kappa_grid[{}] is {}, rho is {}'
                      .format(i, np.array(y_row - y_grid[i]), x_row))

            f_opal = np.append(f_opal, y_grid[i])
            f_rho = np.append(f_rho, sol)

        return np.vstack((f_rho, f_opal))

    @staticmethod
    def case_1(x_row, y_row, y_grid, n_interp):
        f_opal = []
        f_rho = []
        for i in range(len(y_grid)):

            sol = Math.solv_inter_row(x_row, y_row, y_grid[i])

            if (len(sol) == 1):
                f_opal = np.append(f_opal, y_grid[i])
                f_rho = np.append(f_rho, sol)

            if (len(sol) == 2):
                f_opal = np.append(f_opal, y_grid[i])
                f_rho = np.append(f_rho, sol[1])  # attach second element

            if (len(sol) > 2):
                f_opal = np.append(f_opal, y_grid[i])
                f_rho = np.append(f_rho, sol[len(sol) - 1])  # the last element

        return np.vstack((f_rho, f_opal))

    @staticmethod
    def case_2(x_row, y_row, y_grid, n_interp):
        f_opal = []
        f_rho = []
        for i in range(len(y_grid)):

            sol = Math.solv_inter_row(x_row, y_row, y_grid[i])

            if (len(sol) == 2):
                f_opal = np.append(f_opal, y_grid[i])
                f_rho = np.append(f_rho, sol[0])  # attach first(!) element

            if (len(sol) == 1):  # should be just one element.
                f_opal = np.append(f_opal, y_grid[i])
                f_rho = np.append(f_rho, sol)

        return np.vstack((f_rho, f_opal))

    @staticmethod
    def case_3(x_row, y_row, y_grid, n_interp):
        f_opal = []
        f_rho = []

        for i in range(len(y_grid)):
            sol = Math.solv_inter_row(x_row, y_row, y_grid[i])

            if (len(sol) == 1):  # should be just one element.
                f_opal = np.append(f_opal, y_grid[i])
                f_rho = np.append(f_rho, sol)

            if (len(sol) == 2):
                f_opal = np.append(f_opal, y_grid[i])
                f_rho = np.append(f_rho, sol[1])  # attach second(!) element

        return np.vstack((f_rho, f_opal))

    @staticmethod
    def case_4(x_row, y_row, y_grid, n_interp):
        f_opal = []
        f_rho = []
        for i in range(len(y_grid)):
            sol = Math.solv_inter_row(x_row, y_row, y_grid[i])

            if (len(sol) == 1):
                f_opal = np.append(f_opal, y_grid[i])
                f_rho = np.append(f_rho, sol)

            if (len(sol) == 2):
                f_opal = np.append(f_opal, y_grid[i])
                f_rho = np.append(f_rho, sol[0])  # attach first element

        return np.vstack((f_rho, f_opal))

    @staticmethod
    def case_5(x_row, y_row, y_grid, n_interp):
        f_opal = []
        f_rho = []
        for i in range(len(y_grid)):

            sol = Math.solv_inter_row(x_row, y_row, y_grid[i])

            if (len(sol) > 1):
                f_opal = np.append(f_opal, y_grid[i])
                f_rho = np.append(f_rho, sol[-1])

            if (len(sol) == 1):
                f_opal = np.append(f_opal, y_grid[i])
                f_rho = np.append(f_rho, sol)

        print('\t__Note: In case_5 the 3rd out of 3 solutions is used. UNSAFE!')
        return np.vstack((f_rho, f_opal))

    @staticmethod
    def solve_for_row(lim_y1, lim_y2, case, n_interp, x_row, y_row):
        # After the Case has been identified, the actual (T=const, kappa[])-> rho[] can be done
        # given:
        # lim_op1 = op1#np.log10(0.54) #for 20sm
        # lim_op2 = op2#np.log10(0.94) #for 10sm model
        # depth = 1000

        kappa_grid = np.mgrid[lim_y1:lim_y2:n_interp * 1j]

        # treat cases:
        if (case == 0):  # monotonically increasing array - only one solution
            return Row_Analyze.case_0(x_row, y_row, kappa_grid, n_interp)

        if (case == 1):  # there is a decreasing part in the beginning, small.
            return Row_Analyze.case_1(x_row, y_row, kappa_grid, n_interp)

        if (case == 2):
            return Row_Analyze.case_2(x_row, y_row, kappa_grid, n_interp)

        if (case == 3):
            return Row_Analyze.case_3(x_row, y_row, kappa_grid, n_interp)

        if (case == 4):  # there is a decreasing part in the end, small.
            return Row_Analyze.case_4(x_row, y_row, kappa_grid, n_interp)

        if (case == 5):  # there are only single solutions and one triple! (!)
            return Row_Analyze.case_5(x_row, y_row, kappa_grid, n_interp)

        print('\t___Error! Case unspecified! | solve_for_row | case:', case)


class PhysPlots:
    def __init__(self):
        pass

    @staticmethod
    def xy_profile(nm_x, nm_y, x1, y1, lx = np.zeros(1,), ly = np.zeros(1,),
                    x2=np.zeros(1,),y2=np.zeros(1,),x3=np.zeros(1,),y3=np.zeros(1,),x4=np.zeros(1,),
                    y4=np.zeros(1,),x5=np.zeros(1,),y5=np.zeros(1,),x6=np.zeros(1,),y6=np.zeros(1,),
                    x7=np.zeros(1, ), y7=np.zeros(1, )):

        plot_name = './results/' + nm_x + '_' + nm_y + 'profile.pdf'

        # plot_name = 'Vel_profile.pdf'

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)

        plt.title('Velocity Profile')

        plt.plot(x1, y1, '-', color='blue', label='model_1')

        if x2.shape!=(1,) and y2.shape!=(1,):
            plt.plot(x2, y2, '-', color='cyan', label='model_2')

        if x3.shape!=(1,) and y3.shape!=(1,):
            plt.plot(x3, y3, '-', color='green', label='model_3')

        if x4.shape!=(1,) and y4.shape!=(1,):
            plt.plot(x4, y4, '-', color='yellow', label='model_4')

        if x5.shape!=(1,) and y5.shape!=(1,):
            plt.plot(x5, y5, '-', color='orange', label='model_5')

        if x6.shape!=(1,) and y6.shape!=(1,):
            plt.plot(x6, y6, '-', color='red', label='model_6')

        if x7.shape!=(1,) and y7.shape!=(1,):
            plt.plot(x7, y7, '-', color='purple', label='model_6')

        plt.xlabel(nm_x)
        plt.ylabel(nm_y)

        #---------------------------------------MINOR-TICKS-------------------------------
        if lx.shape != (1,):
            major_xticks = np.arange(lx[0], lx[-1] + 1, (lx[-1] -lx[0]) / 5)
            minor_xticks = np.arange(lx[0], lx[-1], (lx[-1] -lx[0]) / 10)
            ax.set_xticks(major_xticks)
            ax.set_xticks(minor_xticks, minor=True)
        # else:
        #     major_xticks = np.arange(x1[0], x1[-1] , (x1[-1] - x1[0]) / 5)
        #     minor_xticks = np.arange(x1[0], x1[-1], (x1[-1] - x1[0]) / 10)

        if ly.shape != (1,):
            major_yticks = np.arange(ly[0], ly[-1] + 1, (ly[-1] -ly[0]) / 5)
            minor_yticks = np.arange(ly[0], ly[-1], (ly[-1] -ly[0]) / 10)
            ax.set_yticks(major_yticks)
            ax.set_yticks(minor_yticks, minor=True)
        # else:
        #     major_yticks = np.arange(y1.min(), y1.max() + 1, (y1.max() - y1.min()) / 5)
        #     minor_yticks = np.arange(y1.min(), y1.min(), (y1.max() - y1.min()) / 10)

        # ax.set_xticks(major_xticks)
        # ax.set_xticks(minor_xticks, minor=True)
        # ax.set_yticks(major_yticks)
        # ax.set_yticks(minor_yticks, minor=True)


        #-------------------------------------VERT/HORISONTAL LINES------------------------------
        # if lim_k1 != None:
        #     lbl = 'k1: ' + str("%.2f" % lim_k1)
        #     plt.axhline(y=lim_k1, color='r', linestyle='dashed', label=lbl)
        #
        # if lim_k2 != None:
        #     lbl = 'k1: ' + str("%.2f" % lim_k2)
        #     plt.axhline(y=lim_k2, color='r', linestyle='dashed', label=lbl)
        #
        # if lim_t1 != None:
        #     lbl = 't1: ' + str("%.2f" % lim_t1)
        #     plt.axvline(x=lim_t1, color='r', linestyle='dashed', label=lbl)
        #
        # if lim_t2 != None:
        #     lbl = 't2: ' + str("%.2f" % lim_t2)
        #     plt.axvline(x=lim_t2, color='r', linestyle='dashed', label=lbl)
        #
        # if it1 != None:
        #     lbl = 'int t1: ' + str("%.2f" % it1)
        #     plt.axvline(x=it1, color='orange', linestyle='dashed', label=lbl)
        #
        # if it2 != None:
        #     lbl = 'int t2: ' + str("%.2f" % it2)
        #     plt.axvline(x=it2, color='orange', linestyle='dashed', label=lbl)


        #----------------------------BOXES------------------------------
        # if any(y2_arr):
        #     ax.fill_between(x_arr, y_arr, y2_arr, label ='Available Region')
        #
        # if it1 != None and it2 != None and lim_k1 != None and lim_k2 != None:
        #     ax.fill_between(np.array([it1, it2]), np.array([lim_k1]), np.array([lim_k2]), label='Interpolation Region')
        #
        # if lim_t1 != None and lim_t2 != None and lim_k1 != None and lim_k2 != None:
        #     ax.fill_between(np.array([lim_t1, lim_t2]), np.array([lim_k1]), np.array([lim_k2]), label='Selected Region')


        ax.grid(which='both')
        ax.grid(which='minor', alpha=0.2)
        ax.grid(which='major', alpha=0.2)

        plt.legend(bbox_to_anchor=(1, 1), loc='upper right', ncol=1)


        plt.savefig(plot_name)


        plt.show()

    @staticmethod
    def xy_2y_profile(nm_x, nm_y, nm_yy, x1, y1, y11,
                    x2=np.zeros(1,), y2=np.zeros(1,), y22=np.zeros(1,),
                    x3=np.zeros(1,), y3=np.zeros(1,), y33=np.zeros(1,),
                    x4=np.zeros(1,), y4=np.zeros(1,), y44=np.zeros(1,),
                    x5=np.zeros(1,), y5=np.zeros(1,), y55=np.zeros(1,),
                    x6=np.zeros(1,), y6=np.zeros(1,), y66=np.zeros(1,),):
        '''***************************WITH-T-as-X-AXIS-------------------------------'''
        fig, ax1 = plt.subplots()

        ax1.plot(x1, y1, '-', color='blue', label='model_1')

        if x2.shape!=(1,) and y2.shape!=(1,):
            ax1.plot(x2, y2, '-', color='cyan', label='model_2')

        if x3.shape!=(1,) and y3.shape!=(1,):
            ax1.plot(x3, y3, '-', color='green', label='model_3')

        if x4.shape!=(1,) and y4.shape!=(1,):
            ax1.plot(x4, y4, '-', color='yellow', label='model_4')

        if x5.shape!=(1,) and y5.shape!=(1,):
            ax1.plot(x5, y5, '-', color='orange', label='model_5')

        if x6.shape!=(1,) and y6.shape!=(1,):
            ax1.plot(x6, y6, '-', color='red', label='model_6')


        # ax1.plot(t2ph, ro2ph, 'gray')
        # ax1.plot(t3ph, ro3ph, 'gray')
        # ax1.plot(t4ph, ro4ph, 'gray')
        # ax1.plot(t5ph, ro5ph, 'gray')
        # ax1.plot(t6ph, ro6ph, 'gray')
        #
        # ax1.plot(t1, ro1, 'b-')
        # ax1.plot(last_elmt(t1), last_elmt(ro1), 'bo')
        # ax1.plot(t2, ro2, 'b-')
        # ax1.plot(last_elmt(t2), last_elmt(ro2), 'bo')
        # ax1.plot(t3, ro3, 'b-')
        # ax1.plot(last_elmt(t3), last_elmt(ro3), 'bo')
        # ax1.plot(t4, ro4, 'b-')
        # ax1.plot(last_elmt(t4), last_elmt(ro4), 'bo')
        # ax1.plot(t5, ro5, 'b-')
        # ax1.plot(last_elmt(t5), last_elmt(ro5), 'bo')
        # ax1.plot(t6, ro6, 'b-')
        # ax1.plot(last_elmt(t6), last_elmt(ro6), 'bo')

        ax1.set_xlabel(nm_x)
        # Make the y-axis label, ticks and tick labels match the line color.
        ax1.set_ylabel(nm_y, color='b')
        ax1.tick_params('y', colors='b')
        ax1.set_xlim(6.2, 4.6)
        plt.grid()

        ax2 = ax1.twinx()

        # ----------------------------EDDINGTON OPACITY------------------------------------
        # ax2.plot(np.mgrid[x1.min():x1.max():100j], np.mgrid[edd_k:edd_k:100j], c='black')


        ax2.plot(x1, y11, '--', color='blue', label='model_1')

        if x2.shape!=(1,) and y22.shape!=(1,):
            ax2.plot(x2, y22, '--', color='cyan', label='model_2')

        if x3.shape!=(1,) and y33.shape!=(1,):
            ax2.plot(x3, y33, '--', color='green', label='model_3')

        if x4.shape!=(1,) and y44.shape!=(1,):
            ax2.plot(x4, y44, '--', color='yellow', label='model_4')

        if x5.shape!=(1,) and y55.shape!=(1,):
            ax2.plot(x5, y55, '--', color='orange', label='model_5')

        if x6.shape!=(1,) and y6.shape!=(1,):
            ax2.plot(x6, y66, '--', color='red', label='model_6')


        # ax2.plot(t1ph, k1ph, 'gray')
        # ax2.plot(t2ph, k2ph, 'gray')
        # ax2.plot(t3ph, k3ph, 'gray')
        # ax2.plot(t4ph, k4ph, 'gray')
        # ax2.plot(t5ph, k5ph, 'gray')
        # ax2.plot(t6ph, k6ph, 'gray')
        #
        # ax2.plot(t1, k1, 'r-')
        # ax2.plot(t2, k2, 'r-')
        # ax2.plot(t3, k3, 'r-')
        # ax2.plot(t4, k4, 'r-')
        # ax2.plot(t5, k5, 'r-')
        # ax2.plot(t6, k6, 'r-')

        ax2.set_ylabel(nm_yy, color='r')
        ax2.tick_params('y', colors='r')

        plt.axvline(x=4.6, color='black', linestyle='solid', label='T = 4.6, He Op Bump')
        plt.axvline(x=5.2, color='black', linestyle='solid', label='T = 5.2, Fe Op Bump')
        plt.axvline(x=6.2, color='black', linestyle='solid', label='T = 6.2, Deep Fe Op Bump')

        # plt.ylim(-8.5, -4)
        fig.tight_layout()
        plt.legend(bbox_to_anchor=(0, 1), loc='upper left', ncol=1)
        plt.show()

    @staticmethod
    def Rho_k_plot(f_rho, f_kappa, rho_row = None, kappa_row = None,
                     lim_k1=None, lim_k2=None, case=None, temp=None, step=None, plot_dir = '../data/plot/'):

        plot_name = 'Rho_k_plot.pdf'
        # path = '/media/vnedora/HDD/opal_anal4/'
        path = plot_dir

        if (temp == None and step == None):
            plot_name = ''.join([path, 'plot_one_raw.pdf'])
        if (temp != None and step == None):
            plot_name = ''.join([path, 'T=', str("%.2f" % temp), '.pdf'])
        if (temp != None and step != None):
            plot_name = ''.join([path, str(step), '_T=', str("%.2f" % temp), '.pdf'])

        # Title of the file Cases
        plot_title = 'Rho_k_plot.pdf'
        if (temp == None and step == None):
            plot_title = ''.join(['T = const'])
        if (temp != None and step == None):
            plot_title = ''.join(['T = ', str(temp)])
        if (temp != None and step != None):
            plot_title = ''.join(['T(', str(step), ') = ', str(temp)])

        # case lalbe
        label_case = ''
        if case != None:
            label_case = ''.join(['Case: ', str(case)])

        # x coordinates of the selected region:
        rho1 = f_rho[Math.find_nearest_index(f_kappa, lim_k1)]
        rho2 = f_rho[Math.find_nearest_index(f_kappa, lim_k2)]

        # labels for vertical an horisontal lines
        lbl_rho_lim = ''
        lbl_op_lim = ''
        if lim_k1 != None and lim_k2 != None:
            lbl_rho_lim = ''.join(['Selected dencity(', str("%.2f" % rho1), ' ', str("%.2f" % rho2), ')'])
            lbl_op_lim = ''.join(['Selected opacity(', str("%.2f" % lim_k1), ' ', str("%.2f" % lim_k2), ')'])

        # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<PLOT>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)

        plt.title(plot_title)
        # pl.ylim(-4, 4)
        # pl.xlim(-10, 10)

        plt.plot(f_rho, f_kappa, '.',color='blue', label='(T, kap=[])->rho[]')
        if any(rho_row) and any(kappa_row):
            plt.plot(rho_row, kappa_row, 'x', color='black', label='table')

        if lim_k1 != None and lim_k2 != None:
            plt.axvspan(rho1, rho2, color='lightblue', linestyle='dotted', label=lbl_rho_lim)
            plt.axhspan(lim_k1, lim_k2, color='lightblue', linestyle='dotted', label=lbl_op_lim)

            plt.axvline(x=rho1, color='grey', linestyle='dotted')
            plt.axvline(x=rho2, color='grey', linestyle='dotted')

            plt.axhline(y=lim_k1, color='grey', linestyle='dotted')
            plt.axhline(y=lim_k2, color='grey', linestyle='dotted')

        plt.xlabel('log(rho)')
        plt.ylabel('opacity')

        if case != None:
            ax.text(f_rho.min(), f_kappa.mean(), label_case, style='italic',
                bbox={'facecolor': 'yellow', 'alpha': 0.5, 'pad': 10})
            # box with data about case and limits of kappa

        if any(rho_row) and any(kappa_row):
            major_xticks = np.arange(rho_row.min(), rho_row.max() + 1, 1)
            minor_xticks = np.arange(rho_row.min(), rho_row.max(), 0.5)
            major_yticks = np.arange(kappa_row.min(), kappa_row.max() + 1, ((kappa_row.max() - kappa_row.min()) / 4))
            minor_yticks = np.arange(kappa_row.min(), kappa_row.max(), ((kappa_row.max() - kappa_row.min()) / 8))
        else:
            major_xticks = np.arange(f_rho.min(), f_rho.max() + 1, 1)
            minor_xticks = np.arange(f_rho.min(), f_rho.max(), 0.5)
            major_yticks = np.arange(f_kappa.min(), f_kappa.max() + 1, ((f_kappa.max() - f_kappa.min()) / 4))
            minor_yticks = np.arange(f_kappa.min(), f_kappa.max(), ((f_kappa.max() - f_kappa.min()) / 8))


        ax.set_xticks(major_xticks)
        ax.set_xticks(minor_xticks, minor=True)
        ax.set_yticks(major_yticks)
        ax.set_yticks(minor_yticks, minor=True)

        ax.grid(which='both')
        ax.grid(which='minor', alpha=0.2)
        ax.grid(which='major', alpha=0.2)

        plt.legend(bbox_to_anchor=(0, 1), loc='upper left', ncol=1)


        plt.savefig(plot_name)

        plt.show()

    @staticmethod
    def k_vs_t(t_arr, y_arr, y2_arr, show = False, save = False,
               lim_k1 = None, lim_k2 = None, lim_t1 = None, lim_t2 = None, it1 = None, it2 = None, plot_dir = '../data/plots/'):

        # plot_name = './results/Kappa_Limits.pdf'

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)

        print(t_arr, '\n', y_arr)
        if len(t_arr) != len(y_arr):
            print('\t Error. len(t_arr {}) != len(y_arr {})'.format(len(t_arr), len(y_arr)))
            print('\t t_arr: {}'.format(t_arr))
            print('\t y_arr: {}'.format(y_arr))
            raise ValueError

        if y2_arr.any() and len(t_arr) != len(y2_arr):
            print('\t Error. len(t_arr {}) != len(y2_arr {})'.format(len(t_arr), len(y2_arr)))
            print('\t t_arr {}:'.format(t_arr))
            print('\t y_arr: {}'.format(y2_arr))
            raise ValueError

        plt.title('Limit Kappa = f(Temperature)')
        plt.plot(t_arr, y_arr, '-', color='blue', label='min k')
        if any(y2_arr):
            plt.plot(t_arr, y2_arr, '-', color='red', label='max k')

        plt.xlabel('t')
        plt.ylabel('kappa')

        #---------------------------------------MINOR-TICKS-------------------------------
        if it1 != None and it2 != None and lim_t1 != None and lim_t2 != None:
            major_xticks = np.array([t_arr.min(), lim_t1, it1, it2, lim_t2, t_arr.max()])
        else:
            major_xticks = np.array([t_arr.min(), t_arr.max()])
        minor_xticks = np.arange(t_arr.min(), t_arr.max(), 0.2)

        #---------------------------------------MAJOR TICKS-------------------------------
        major_yticks = np.arange(y_arr.min(), y_arr.max() + 1, ((y_arr.max() - y_arr.min()) / 4))
        minor_yticks = np.arange(y_arr.min(), y_arr.max(), ((y_arr.max() - y_arr.min()) / 8))

        if any(y2_arr):
            major_yticks = np.arange(y_arr.min(), y2_arr.max() + 1, ((y2_arr.max() - y_arr.min()) / 4))
            minor_yticks = np.arange(y_arr.min(), y2_arr.max(), ((y2_arr.max() - y_arr.min()) / 8))

        if any(y2_arr) and lim_k1 !=None and  lim_k2 != None:
            major_yticks = np.array([y_arr.min(), lim_k1, lim_k2, y2_arr.max()])
            minor_yticks = np.arange(y_arr.min(), y2_arr.max(), ((y2_arr.max() - y_arr.min()) / 10))


        ax.set_xticks(major_xticks)
        ax.set_xticks(minor_xticks, minor=True)
        ax.set_yticks(major_yticks)
        ax.set_yticks(minor_yticks, minor=True)

        #-------------------------------------VERT/HORISONTAL LINES------------------------------
        if lim_k1 != None:
            lbl = 'k1: ' + str("%.2f" % lim_k1)
            plt.axhline(y=lim_k1, color='r', linestyle='dashed',
                        label='k1:{} lm1:{} l:{}'.format("%.2f" % lim_k1,
                                                    "%.2f" % Physics.logk_loglm(lim_k1),
                                                    "%.2f" % Physics.lm_to_l_langer(Physics.logk_loglm(lim_k1))))

        if lim_k2 != None:
            plt.axhline(y=lim_k2, color='r', linestyle='dashed',
                        label='k2:{} lm2:{} l:{}'.format("%.2f" % lim_k2,
                                                         "%.2f" % Physics.logk_loglm(lim_k2),
                                                         "%.2f" % Physics.lm_to_l_langer(Physics.logk_loglm(lim_k2))))

        if lim_t1 != None:
            lbl = 't1: ' + str("%.2f" % lim_t1)
            plt.axvline(x=lim_t1, color='r', linestyle='dashed', label=lbl)

        if lim_t2 != None:
            lbl = 't2: ' + str("%.2f" % lim_t2)
            plt.axvline(x=lim_t2, color='r', linestyle='dashed', label=lbl)

        if it1 != None:
            lbl = 'int t1: ' + str("%.2f" % it1)
            plt.axvline(x=it1, color='orange', linestyle='dashed', label=lbl)

        if it2 != None:
            lbl = 'int t2: ' + str("%.2f" % it2)
            plt.axvline(x=it2, color='orange', linestyle='dashed', label=lbl)


        #----------------------------BOXES------------------------------
        if any(y2_arr):
            ax.fill_between(t_arr, y_arr, y2_arr, label = 'Available Region')

        if it1 != None and it2 != None and lim_k1 != None and lim_k2 != None:
            ax.fill_between(np.array([it1, it2]), np.array([lim_k1]), np.array([lim_k2]), label='Interpolation Region')

        if lim_t1 != None and lim_t2 != None and lim_k1 != None and lim_k2 != None:
            ax.fill_between(np.array([lim_t1, lim_t2]), np.array([lim_k1]), np.array([lim_k2]), label='Selected Region')


        ax.grid(which='both')
        ax.grid(which='minor', alpha=0.2)
        ax.grid(which='major', alpha=0.2)

        plt.legend(bbox_to_anchor=(1, 1), loc='upper right', ncol=1)

        if save:
            plot_name = plot_dir+'Kappa_Limits.pdf'
            plt.savefig(plot_name)

        if show:
            plt.show()

    # @staticmethod
    # def rho_vs_t(t_arr, y_arr):
    #
    #     plot_name = 'Rho_t_for_a_kappa.pdf'
    #
    #     fig = plt.figure()
    #     ax = fig.add_subplot(1, 1, 1)
    #
    #     plt.title('Rho = f(Temperature) for one kappa')
    #     plt.plot(t_arr, y_arr, '-', color='blue', label='k1')
    #
    #     plt.ylim(y_arr.min(), y_arr.max())
    #     plt.xlim(t_arr.min(), t_arr.max())
    #
    #     plt.xlabel('t')
    #     plt.ylabel('rho')
    #
    #     major_xticks = np.arange(t_arr.min(), t_arr.max()+0.1, (t_arr.max() - t_arr.min())/4)
    #     minor_xticks = np.arange(t_arr.min(), t_arr.max(), (t_arr.max() - t_arr.min())/8)
    #
    #     major_yticks = np.arange(y_arr.min(), y_arr.max() + 0.1, ((y_arr.max() - y_arr.min()) / 4))
    #     minor_yticks = np.arange(y_arr.min(), y_arr.max(), ((y_arr.max() - y_arr.min()) / 8))
    #
    #
    #     ax.set_xticks(major_xticks)
    #     ax.set_xticks(minor_xticks, minor=True)
    #     ax.set_yticks(major_yticks)
    #     ax.set_yticks(minor_yticks, minor=True)
    #
    #     ax.grid(which='both')
    #     ax.grid(which='minor', alpha=0.2)
    #     ax.grid(which='major', alpha=0.2)
    #
    #     plt.legend(bbox_to_anchor=(1, 1), loc='upper right', ncol=1)
    #
    #     plt.savefig(plot_name)
    #
    #
    #     plt.show()

    @staticmethod
    def t_rho_kappa(t, rho, kappa, edd_1 = np.zeros((1,)),
                    m_t = np.zeros((1,)), m_rho =  np.zeros((1,))):

        name = './results/t_rho_kappa.pdf'
        plt.figure()


        levels = [0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]

        plt.xlim(t.min(), t.max())
        plt.ylim(rho.min(), rho.max())
        contour_filled = plt.contourf(t, rho, 10 ** (kappa), levels)
        plt.colorbar(contour_filled)
        contour = plt.contour(t, rho, 10 ** (kappa), levels, colors='k')
        plt.clabel(contour, colors='k', fmt='%2.1f', fontsize=12)
        plt.title('OPACITY PLOT')
        plt.xlabel('Log(T)')
        plt.ylabel('log(rho)')
        plt.axvline(x=4.2, color='r', linestyle='dashed', label='HeI Bump')
        plt.axvline(x=4.6, color='r', linestyle='dashed', label='HeII Fe Bump')
        plt.axvline(x=5.2, color='r', linestyle='dashed', label='Fe Bump')
        plt.axvline(x=6.2, color='r', linestyle='dashed', label='Deep Fe Bump')
        # plt.axhline(y = vrho, color='r', linestyle='dashed', label = lbl2)
        # pl.plot(t_edd, rho_edd, marker='o', color = 'r')
        if edd_1.any():
            plt.plot(edd_1[0, :], edd_1[1, :], '-', color='w')
        # if edd_2.any():
        #     pl.plot(edd_2[0, :], edd_2[1, :], '-', color='w')
        # if edd_3.any():
        #     pl.plot(edd_3[0, :], edd_3[1, :], '-', color='w')

        if m_rho.any() and m_t.any():
            plt.plot(m_t, m_rho, '-', color='maroon')
            plt.plot(m_t[-1], m_rho[-1], 'o', color='maroon')

        # proc=subprocess.Popen(shlex.split('lpr {f}'.format(f=fname)))
        # plt.legend()


        plt.savefig(name)

        plt.show()

    @staticmethod
    def t_kappa_rho(t, kappa, rho2d, mins=None, p1_t = None, p1_lm = None, val1_mdot = None,
                    p2_t = None, p2_lm = None, val2_mdot = None,
                    p3_t=None, p3_lm=None, val3_mdot=None,
                    p4_t=None, p4_lm=None, val4_mdot=None):

        name = './results/t_LM_Mdot_plot.pdf'

        plt.figure()

        # if new_levels != None:
        #     levels = new_levels
        # else:
        #     levels = [-8, -7, -6, -5, -4, -3, -2]

        plt.xlim(t.min(), t.max())
        plt.ylim(kappa.min(), kappa.max())
        levels = [-7.5, -7, -6.5, -6, -5.5, -5, -4.5, -4, -3.5, -3, -2.5, -2]
        #levels = [-10, -9, -8, -7, -6, -5, -4]
        contour_filled = plt.contourf(t, kappa, rho2d.T, levels)
        plt.colorbar(contour_filled)
        contour = plt.contour(t, kappa, rho2d.T, levels, colors='k')
        plt.clabel(contour, colors='k', fmt='%2.1f', fontsize=12)
        plt.title('MASS LOSS PLOT')
        plt.xlabel('Log(t)')
        plt.ylabel('log(L/M)')
        # plt.axvline(x=4.2, color='r', linestyle='dashed', label='HeI Bump')
        # plt.axvline(x=4.6, color='r', linestyle='dashed', label='HeII Fe Bump')
        # plt.axvline(x=5.2, color='r', linestyle='dashed', label='Fe Bump')
        # plt.axvline(x=6.2, color='r', linestyle='dashed', label='Deep Fe Bump')
        # if p1_t != None and p1_lm != None:
        #     plt.axvline(x=p1_t, color='w', linestyle='dashed', label='p_t: {}, p_L/M: {}'.format("%.2f" % p1_t, "%.2f" % p1_lm))
        #     plt.axhline(y=p1_lm, color='w', linestyle='dashed', label='Expected M_dot: {}'.format("%.2f" % val1_mdot))

        plt.plot(mins[0,:], mins[1,:], '-', color='blue', label='min Mdot')

        if p1_t != None and p1_lm != None and val1_mdot != None:
            plt.plot([p1_t], [p1_lm], marker='x', markersize=9, color="blue",
                     label='Model 1: T_s {} , L/M {} , Mdot {}'.format(p1_t, "%.2f" % p1_lm, "%.2f" % val1_mdot))

        if p2_t != None and p2_lm != None and val2_mdot != None:
            plt.plot([p2_t], [p2_lm], marker='x', markersize=9, color="cyan",
                     label='Model 1: T_s {} , L/M {} , Mdot {}'.format(p2_t, "%.2f" % p2_lm, "%.2f" % val2_mdot))

        if p3_t != None and p3_lm != None and val3_mdot != None:
            plt.plot([p3_t], [p3_lm], marker='x', markersize=9, color="magenta",
                     label='Model 1: T_s {} , L/M {} , Mdot {}'.format(p3_t, "%.2f" % p3_lm, "%.2f" % val3_mdot))

        if p4_t != None and p4_lm != None and val4_mdot != None:
            plt.plot([p4_t], [p4_lm], marker='x', markersize=9, color="red",
                     label='Model 2: T_s {} , L/M {} , Mdot {}'.format(p4_t, "%.2f" % p4_lm, "%.2f" % val4_mdot))

        # plt.axhline(y = vrho, color='r', linestyle='dashed', label = lbl2)
        # pl.plot(t_edd, rho_edd, marker='o', color = 'r')
        # if edd_1.any():
        #     pl.plot(edd_1[0, :], edd_1[1, :], '-', color='w')
        # if edd_2.any():
        #     pl.plot(edd_2[0, :], edd_2[1, :], '-', color='w')
        # if edd_3.any():
        #     pl.plot(edd_3[0, :], edd_3[1, :], '-', color='w')


        # proc=subprocess.Popen(shlex.split('lpr {f}'.format(f=fname)))
        plt.legend()


        plt.savefig(name)

        plt.show()


    # @staticmethod
    # def t_kappa_rho(t, kappa, rho2d, new_levels = None, save = True):
    #
    #     plt.figure()
    #
    #     # if new_levels != None:
    #     #     levels = new_levels
    #     # else:
    #     #     levels = [0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
    #
    #     pl.xlim(t.min(), t.max())
    #     pl.ylim(kappa.min(), kappa.max())
    #     contour_filled = plt.contourf(t, kappa, rho2d.T)
    #     plt.colorbar(contour_filled)
    #     contour = plt.contour(t, kappa, rho2d.T, colors='k')
    #     plt.clabel(contour, colors='k', fmt='%2.1f', fontsize=12)
    #     plt.title('DENSITY PLOT')
    #     plt.xlabel('Log(t)')
    #     plt.ylabel('log(kappa)')
    #     plt.axvline(x=4.2, color='r', linestyle='dashed', label='HeI Bump')
    #     plt.axvline(x=4.6, color='r', linestyle='dashed', label='HeII Fe Bump')
    #     plt.axvline(x=5.2, color='r', linestyle='dashed', label='Fe Bump')
    #     plt.axvline(x=6.2, color='r', linestyle='dashed', label='Deep Fe Bump')
    #     # plt.axhline(y = vrho, color='r', linestyle='dashed', label = lbl2)
    #     # pl.plot(t_edd, rho_edd, marker='o', color = 'r')
    #     # if edd_1.any():
    #     #     pl.plot(edd_1[0, :], edd_1[1, :], '-', color='w')
    #     # if edd_2.any():
    #     #     pl.plot(edd_2[0, :], edd_2[1, :], '-', color='w')
    #     # if edd_3.any():
    #     #     pl.plot(edd_3[0, :], edd_3[1, :], '-', color='w')
    #
    #
    #     # proc=subprocess.Popen(shlex.split('lpr {f}'.format(f=fname)))
    #     # plt.legend()
    #     fname = 'k_t_rho_plot.pdf'
    #
    #     plt.savefig(fname)
    #
    #     plt.show()

    @staticmethod
    def t_mdot_lm(t, mdot, lm, p1_t = None, p1_mdot = None, p1_lm = None,
                  p2_t = None, p2_mdot = None, p2_lm = None,
                  p3_t = None, p3_mdot = None, p3_lm = None,
                  p4_t = None, p4_mdot = None, p4_lm = None):
        name = './results/t_mdot_lm_plot.pdf'

        plt.figure()

        # if new_levels != None:
        #     levels = new_levels
        # else:
        #     levels = [-8, -7, -6, -5, -4, -3, -2]

        plt.xlim(t.min(), t.max())
        plt.ylim(mdot.min(), mdot.max())
        levels = [3.0, 3.2, 3.4, 3.6, 3.8, 4.0, 4.1, 4.2, 4.3, 4.6, 4.8, 5.0, 5.2]
        contour_filled = plt.contourf(t, mdot, lm, levels)
        plt.colorbar(contour_filled)
        contour = plt.contour(t, mdot, lm, levels, colors='k')
        plt.clabel(contour, colors='k', fmt='%2.1f', fontsize=12)
        plt.title('L/M PLOT')
        plt.xlabel('Log(t_s)')
        plt.ylabel('log(M_dot)')
        # plt.axvline(x=4.2, color='r', linestyle='dashed', label='HeI Bump')
        # plt.axvline(x=4.6, color='r', linestyle='dashed', label='HeII Fe Bump')
        # plt.axvline(x=5.2, color='r', linestyle='dashed', label='Fe Bump')
        # plt.axvline(x=6.2, color='r', linestyle='dashed', label='Deep Fe Bump')

        # if test_t1 != None and test_mdot1 != None and test_lm1 != None:
        #     plt.axvline(x=test_t1, color='c', linestyle='dashed', label='T_s: {} , mdot: {}'.format(test_t1, "%.2f" % test_mdot1))
        #     plt.axhline(y=test_mdot1, color='c', linestyle='dashed',
        #                 label='Star L/M: {}'.format("%.2f" % test_lm1))
        #
        # if test_t2 != None and test_mdot2 != None and test_lm2 != None:
        #     plt.axvline(x=test_t2, color='m', linestyle='dashed', label='T_s: {} , mdot: {}'.format(test_t2, "%.2f" % test_mdot2))
        #     plt.axhline(y=test_mdot2, color='m', linestyle='dashed',
        #                 label='Star L/M: {}'.format("%.2f" % test_lm2))

        if p1_t != None and p1_mdot != None and p1_lm != None:
            plt.plot([p1_t], [p1_mdot], marker='x', markersize=9, color="blue",
                     label='Model 1: T_s {} , mdot {} , L/M: {}'.format(p1_t, "%.2f" % p1_mdot, "%.2f" % p1_lm))

        if p2_t != None and p2_mdot != None and p2_lm != None:
            plt.plot([p2_t], [p2_mdot], marker='x', markersize=9, color="cyan",
                     label='Model 2: T_s {} , mdot {} , L/M {}'.format(p2_t, "%.2f" % p2_mdot, "%.2f" % p2_lm))

        if p3_t != None and p3_mdot != None and p3_lm != None:
            plt.plot([p3_t], [p3_mdot], marker='x', markersize=9, color="magenta",
                     label='Model 1: T_s {} , mdot {} , L/M: {}'.format(p3_t, "%.2f" % p3_mdot, "%.2f" % p3_lm))

        if p4_t != None and p4_mdot != None and p4_lm != None:
            plt.plot([p4_t], [p4_mdot], marker='x', markersize=9, color="red",
                     label='Model 2: T_s {} , mdot {} , L/M {}'.format(p4_t, "%.2f" % p4_mdot, "%.2f" % p4_lm))

        # proc=subprocess.Popen(shlex.split('lpr {f}'.format(f=fname)))
        plt.legend()
        fname = 't_mdot_lm_plot.pdf'

        plt.savefig(name)

        plt.show()

    @staticmethod
    def lm_min_mdot(min_mdot_arr, lm_arr, x1 = None, y1 = None,
                                                            x2 = None, y2 = None,
                                                            x3 = None, y3 = None,
                                                            x4 = None, y4 = None):

        plot_name = './results/Min_Mdot.pdf'

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)

        plt.title('L/M = f(min M_dot)')
        plt.plot(min_mdot_arr, lm_arr, '-', color='blue', label='min k')


#-------------

        plt.ylim(lm_arr.min(), lm_arr.max())
        plt.xlim(min_mdot_arr.min(), min_mdot_arr.max())

        plt.xlabel('log(M_dot)')
        plt.ylabel('log(L/M)')


        major_xticks = np.array([-6.5,-6,-5.5,-5,-4.5,-4,-3.5])
        minor_xticks = np.arange(-7.0,-3.5,0.1)

        major_yticks = np.array([3.8,3.9,4.0,4.1,4.2,4.3,4.4,4.5])
        minor_yticks = np.arange(3.8, 4.5, 0.05)

        # major_xticks = np.arange(min_mdot_arr.min(), min_mdot_arr.max()+0.1, (min_mdot_arr.max() - min_mdot_arr.min())/4)
        # minor_xticks = np.arange(min_mdot_arr.min(), min_mdot_arr.max(), (min_mdot_arr.max() - min_mdot_arr.min())/8)
        #
        # major_yticks = np.arange(lm_arr.min(), lm_arr.max() + 0.1, ((lm_arr.max() - lm_arr.min()) / 4))
        # minor_yticks = np.arange(lm_arr.min(), lm_arr.max(), ((lm_arr.max() - lm_arr.min()) / 8))



        ax.grid(which='major', alpha=0.2)

        plt.legend(bbox_to_anchor=(1, 1), loc='upper right', ncol=1)

        if x1 != None and y1 != None:
            plt.plot([x1], [y1],  marker='x', markersize=9, color="blue",
                     label='Model 1: Mdot {} , L/M {}'.format("%.2f" % x1, "%.2f" % y1))
        if x2 != None and y2 != None:
            plt.plot([x2], [y2],  marker='x', markersize=9, color="cyan",
                     label='Model 2: Mdot {} , L/M {}'.format("%.2f" % x2, "%.2f" % y2))
        if x3 != None and y3 != None:
            plt.plot([x3], [y3],  marker='x', markersize=9, color="magenta",
                     label='Model 3: Mdot {} , L/M {}'.format("%.2f" % x3, "%.2f" % y3))
        if x4 != None and y4 != None:
            plt.plot([x4], [y4],  marker='x', markersize=9, color="red",
                     label='Model 4: Mdot {} , L/M {}'.format("%.2f" % x4, "%.2f" % y4))


        ax.set_xticks(major_xticks)
        ax.set_xticks(minor_xticks, minor=True)
        ax.set_yticks(major_yticks)
        ax.set_yticks(minor_yticks, minor=True)

        ax.grid(which='both')
        ax.grid(which='minor', alpha=0.2)


        ax.fill_between(min_mdot_arr, lm_arr, color="orange", label = 'Mdot < Minimun')


        # if x1 != None and y1 != None:
        #     plt.axvline(x=x1, color='g', linestyle='dashed', label='Model1 10sm')
        #     plt.axhline(y=y1, color='g', linestyle='dashed', label=' ')
        #
        # if x2 != None and y2 != None:
        #     plt.axvline(x=x2, color='g', linestyle='dashed', label='Model2 ')
        #     plt.axhline(y=y2, color='g', linestyle='dashed', label=' ')
        #
        # if x3 != None and y3 != None:
        #     plt.axvline(x=x3, color='g', linestyle='dashed', label='Model3 ')
        #     plt.axhline(y=y3, color='g', linestyle='dashed', label=' ')
        # if x4 != None and y4 != None:
        #     plt.axvline(x=x4, color='r', linestyle='dashed', label='Model3 ')
        #     plt.axhline(y=y4, color='r', linestyle='dashed', label=' ')

        plt.legend(bbox_to_anchor=(0, 1), loc='upper left', ncol=1)
        plt.savefig(plot_name)


        plt.show()


class Table_Analyze(Read_Table):

    # o stands for options -------------

    o_cut_mask =   True # by default the full analythisis is performed
    o_cut_rep =    True
    o_cut_smooth = True

    plot_k_vs_t = True  # plots the available kappa region, the selected and the interpolated areas as well.

    def __init__(self, table_name, n_anal_, load_lim_cases, output_dir = './data/', plot_dir = './plots/'):

        # super().__init__(table_name) # using inheritance instead of calling for an instance, to get the rho, kappa and t
        Read_Table.__init__(self, table_name) # better way of inheritting it... The previous one required me to override something
        # cl1 = Read_Table(table_name)

        self.output_dir = output_dir
        self.plot_dir = plot_dir
        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)
        if not os.path.isdir(plot_dir):
            os.mkdir(plot_dir)


        self.rho2d = Physics.get_rho(self.r, self.t)
        self.kappa2d = self.kappas


        # self.n_out = n_out_
        self.n_anal = n_anal_

        self.min_k = []
        self.max_k = []
        self.cases = []
        self.t_avl = []


        self.get_case_limits_in_table(load_lim_cases)

    def case_limits_in_table(self):
        ''' Finds an array of min and max kappas in the kappa=f(rho), - for each raw in
            The number of raws if given my t1 - t2 range. If t1 = t2 - only one t is considered
            and this function becomes equivalent to 'solve_for_row' from 'Raw_Analyze
        '''

        # if t1 == None:
        #     t1 = self.t.min()
        # if t2 == None:
        #     t2 = self.t.max()
        #
        # start = Math.find_nearest_index(self.t, t1)
        # stop  = Math.find_nearest_index(self.t, t2) + 1 # as the last element is exclusive in [i:j] operation

        print('=============================SEARCHING=FOR=LIMITS=&=CASES============================')
        print('\t__None: Peforming search withing temp range: ', self.t.min(), ' ', self.t.max())

        min_k = []
        max_k = []
        cases = []
        t_avl = []

        for i in range(len(self.t)):
            print('\t <---------------------| t = ',self.t[i],' |--------------------->\n')


            tmp1 = Row_Analyze.clear_row(self.rho2d[i, :], self.kappa2d[i, :], self.o_cut_mask, self.o_cut_rep, self.o_cut_smooth)
            tmp2 = Row_Analyze.get_case_limits(tmp1[0], tmp1[1], self.n_anal)

            min_k = np.append(min_k, tmp2[1])
            max_k = np.append(max_k, tmp2[2])
            cases = np.append(cases, tmp2[0])
            t_avl = np.append(t_avl, self.t[i])

            print('\t <-------------------| END of ', self.t[i], ' |-------------------->\n')


        print('===================================END=OF=SEARCHING==================================')

        # ----------------------------Saving the Table --------------------------
        out = np.vstack((cases, min_k, max_k, t_avl))

        table_name_extension = self.table_name.split('/')[-1]
        table_name = table_name_extension.split('.')[0]
        out_name = self.output_dir + table_name + '.caslim'
        np.savetxt(out_name, out, delimiter='  ', fmt='%1.3e')

        return out

    def get_case_limits_in_table(self, load_lim_cases):
        '''
        Updates the "self.cases self.min_k self.max_k self.t_avl" in the class either via direct computation
        or by loading the special file with 'out_name'
        :return:
        '''
        table_name_extension = self.table_name.split('/')[-1]
        table_name = table_name_extension.split('.')[0]
        out_name = self.output_dir +  table_name + '.caslim' # like ../data/table8.data.caslim

        if load_lim_cases :
            load = np.loadtxt(out_name, dtype=float, delimiter='  ')
            print('\t__Note. Table with cases and limits for opal table: < {} > has been loaded succesfully.'
                  .format(self.table_name))

        else:
            load = self.case_limits_in_table()

        self.cases = np.append(self.cases, load[0, :])
        self.min_k = np.append(self.min_k, load[1, :])
        self.max_k = np.append(self.max_k, load[2, :])
        self.t_avl = np.append(self.t_avl, load[3, :])


        print('\t t_aval: {}'.format(self.t_avl))
        print('\t__N: t({} , {})[{}] and t_aval({}, {})[{}]'
              .format(self.t[0], self.t[-1], len(self.t), self.t_avl[0], self.t_avl[-1], len(self.t_avl)))

        if len(self.t_avl) < len(self.t):
            print('Analyzed t_avl {} < {} t given.'.format(len(self.t_avl), len(self.t)))
            #
            # Possible solution is to cut the t_avl from t, together with rho2d and kappa 2d
            #
            raise ValueError('\t__Error. Analyzed t region (t_avl) < than given t | get_case_limits_in_table |')

        if len(self.min_k) == len(self.max_k) == len(self.cases) == len(self.t_avl) ==  len(self.t):
            print('\t__Note: k limits, cases and available t region are set | get_case_limits_in_table | ')

        print('\t__Note: *case_limits* output: (0--) cases, (1--) min_k , (2--) max_k, (3--) t_avl')
        return load

    # @staticmethod
    def check_t_lim(self, t1, t2):
        if t1 > t2:
            raise ValueError('t1 ({}) > t2 ({})'.format(t1,t2))
            # sys.exit('\t__Error. t1 ({}) > t2 ({}) in |check_t_lim| in |Table_Analyze|')
        # a = Errors.is_a_bigger_b(t1, t2,    '|check_t_lim|', True, ' wrong temp. limits')
        if t2 > self.t[-1]:
            raise ValueError('t2 {} > t[-1] {} '.format(t2, self.t[-1]))
            # sys.exit('\t__Error. |check_t_lim|, t2 {} > t[-1] {} '.format(t2, self.t[-1]))
        if t1 < self.t[0]:
            print('\t: t_array is: ({} , {}) consisting of {} elements' .format(self.t[0], self.t[-1], len(self.t)))
            raise ValueError('t1 {} < t[0] {}'.format(t1, self.t[0]))
            # sys.exit('t__Error. |check_t_lim| t1 {} < t[0] {}'.format(t1, self.t[0]))

    def get_it_lim(self, t1, t2, k1, k2):
        indx_1 =  [i for i in range(len(self.t)) if self.t[i] == t1][0]  #GGGGGGenerator
        indx_2 =  [i for i in range(len(self.t)) if self.t[i] == t2][0]  # taking [0] element as it is a list :(


        new_t = []
        new_t = np.append(new_t, self.t[indx_1+1 : indx_2]) # adding the t1-t2 part
        s = indx_1
        for i in range (indx_1): # goes below t1 untill k1, k2 go outside the min/max k range
            if (k1 >= self.min_k[s] and k1 <= self.max_k[s] and k2 >= self.min_k[s] and k2 <= self.max_k[s]):
                new_t = np.append(new_t, self.t[s])
                s = s - 1
            else:
                break

        s = indx_2
        for i in range (len(self.t) - indx_2):# goes up from t1 untill k1, k2 go outside the min/max k range
            if k1 >= self.min_k[s] and k1 <= self.max_k[s] and k2 >= self.min_k[s] and k2 <= self.max_k[s] :
                new_t = np.append(new_t, self.t[s])
                s = s + 1
            else:
                break

        new_t = np.sort(new_t)

        return new_t

    def check_lim_task(self, t1_, t2_, k1, k2):
        '''

        # :param t1: user's t1
        # :param t2: user's t2
        # :param t_arr: from Table_Analyze
        # :param k1: user's k1
        # :param k2: user's k2
        # :param min_k_arr: from Table_Analyze
        # :param max_k_arr: from Table_Analyze
        # :return: [t1, t2, k1, k2, it1, it2], where it1 and it2 are the t limits for interpolation
        '''

        if t2_ > self.t.max(): raise ValueError('t1 > t_lim.max() |check_lim_task|')
        if t2_ < self.t.min(): raise ValueError('t1 < t_lim.min() |check_lim_task|')

        # indx_1 =  [i for i in range(len(self.t)) if self.t[i] == t1][0]  #GGGGGGenerator
        # indx_2 =  [i for i in range(len(self.t)) if self.t[i] == t2][0] # taking [0] element as it is a list :(

        indx_1 = Math.find_nearest_index(self.t, t1_)
        indx_2 = Math.find_nearest_index(self.t, t2_)

        t1 = self.t[indx_1] # Redefining the t range, in case the given t1 t2 do not equal to one of the value
        t2 = self.t[indx_2] #   in the self.t array

        print('\t__Note! Selected t range is from t:[{},{}] to t:[{},{}].'.format(t1_,t2_,t1,t2))

        lim_k1 = np.array( self.min_k[indx_1:indx_2+1] ).max() # looking for lim_k WITHIN the t1, t2 limit from user
        lim_k2 = np.array( self.max_k[indx_1:indx_2+1] ).min()

        print('\t__Note: lim_k1:',lim_k1, 'lim_k2', lim_k2)

        # Errors.is_a_bigger_b(lim_k1, lim_k2, '|check_k_lim|', True, ' lim_k1 > lim_k2')

        if lim_k1 > lim_k2:
            print('\t t_avl: {}'.format(self.t_avl))
            print('\t min_k: {}'.format(self.min_k))
            print('\t max_k: {}'.format(self.max_k))
            print('\t__Error: lim_k1 {} > {} lim_k2 |check_lim_task| '.format(lim_k1, lim_k2))
            print('!!!! DECREASE THE Y2 LIMIT or INCREASE Y1. THE SELECTED REGION IS BIGGER THAN ABAILABEL !!!!')
            raise ValueError

        #if k1 != None and k2 != None:

        if k1 != None and k1 < lim_k1 :
            raise ValueError('\t__Error: k1 < lim_k1 in the region: {} < {}'.format(k1, lim_k1))

        if k2 != None and k2 > lim_k2 :
            raise ValueError('\t__Error: k2 > lim_k2 in the region: {} > {}'.format(k2, lim_k2))

        if k2 != None and k2 < lim_k1 :
            raise ValueError('\t__Error: k2 < lim_k1 in the region: {} < {}'.format(k2, lim_k1))

        if k1 != None and k1 > lim_k2 :
            raise ValueError('\t__Error: k1 > lim_k2 in the region: {} < {}'.format(k1, lim_k2))


        if k1 == None and k2 == None:
            k1 = lim_k1
            k2 = lim_k2
            it1 = t1
            it2 = t2
            print('\t__Note: k1 and k2 not given. Setting k1, k2 : [', "%.2f" % k1,'', "%.2f" % k2,']')
            print('\t__Note: Interpolation is limited to: t:[',it1,'',it2,'] k:[', "%.2f" % k1,'', "%.2f" % k2,']')
            # it1, it2,  - don't have a meaning!
            return [t1, t2, k1, k2, it1, it2]


        if k1 == None and k2 != None and k2 <= lim_k2:
            #get k1 from 'get_case_limits_in_table'

            k1 = lim_k1  # changing k1
            i_t = self.get_it_lim(t1, t2, k1, k2)
            it1 = i_t.min()
            it2 = i_t.max()

            print('\t__Note: k1 is not given. Setting k1: ', "%.2f" % k1)
            print('\t__Note: Interpolation is extended to: t:[', it1,'',it2,'] k:[', "%.2f" % k1,'', "%.2f" % k2,']')

            return [t1, t2, k1, k2, it1, it2]


        if k1 != None and k2 == None and k1 >= lim_k1 :
            print('\t__Note: k1 and k2 not given. Using the available limits in the region')
            #get k2 from 'get_case_limits_in_table'
            k2 = lim_k2

            i_t = self.get_it_lim(t1, t2, k1, k2)
            it1 = i_t.min()
            it2 = i_t.max()

            print('\t__Note: k2 is not given. Setting k2: ', "%.2f" % k2)
            print('\t__Note: Interpolation is extended to: t:[',it1,'',it2,'] k:[', "%.2f" % k1,'', "%.2f" % k2,']')

            return [t1, t2, k1, k2, it1, it2]


        if k1 >= lim_k1 and k2 <= lim_k2 and k1 == k2 :

            i_t = self.get_it_lim(t1, t2, k1, k2)
            it1 = i_t.min()
            it2 = i_t.max()

            print('\t__Note: k1 = k2, Solving for unique k:', "%.2f" % k1)
            print('\t__Note: Interpolation is extended to: t:[',it1,'',it2,'] k:[', "%.2f" % k1,']')

            return [t1, t2, k1, k2, it1, it2]


        if k1 >= lim_k1 and k2 <= lim_k2 : # k1 and k2 != None

            i_t = self.get_it_lim(t1, t2, k1, k2)
            it1 = i_t.min()
            it2 = i_t.max()

            print('\t__Note: Interpolation is extended to: t:[',it1,'',it2,'] k:[', "%.2f" % k1,'', "%.2f" % k2,']')

            return [t1, t2, k1, k2, it1, it2]



        # return np.array([1, k1, k2]) # last value stands for using a band of common kappas for all rows

    def table_plotting(self, t1 = None, t2 = None, n_out = 1000):
        '''
        No universal kappa limits. Use unique limits for every t.
        :return: set of plots
        n_out: is 1000 by default
        '''
        if t1 == None: t1 = self.t.min()
        if t2 == None: t2 = self.t.max()

        self.check_t_lim(t1,t2)

        # i_1 = [i for i in range(len(self.t)) if self.t[i] == t1][0] # for self.t as rho2d and kappa2d have indexes as t
        # i_2 = [i for i in range(len(self.t)) if self.t[i] == t2][0] + 1

        i_1 = Math.find_nearest_index(self.t, t1)
        i_2 = Math.find_nearest_index(self.t, t2) + 1


        f_kappa = np.zeros((i_2 - i_1, n_out))  # all rows showld be the same!
        f_rho =   np.zeros((i_2 - i_1, n_out))
        t_f = []
        print('====================================INTERPOLATING====================================')

        s = 0
        for i in range(i_1, i_2):
            print('\t <---------------------------| t[',i,'] = ',self.t[i],' |---------------------------->\n')

            tmp1 = Row_Analyze.clear_row(self.rho2d[i, :], self.kappa2d[i, :],
                                         self.o_cut_mask,
                                         self.o_cut_rep,
                                         self.o_cut_smooth)
            tmp2 = Row_Analyze.solve_for_row(self.min_k[i], self.max_k[i], self.cases[i], n_out, tmp1[0], tmp1[1])

            f_rho[s, :] = np.array(tmp2[0])
            f_kappa[s, :] = np.array(tmp2[1])
            t_f = np.append(t_f, self.t[s])

            tmp3 = Row_Analyze.cut_mask_val(self.rho2d[i, :],   # ONLY for plotting
                                            self.kappa2d[i, :])  # I have to cut masked values, or they

            PhysPlots.Rho_k_plot(tmp2[0], tmp2[1], tmp3[0], tmp3[1],  # screw up the scale of the plot :(
                                 self.min_k[i], self.max_k[i], self.cases[i], self.t[i], i,
                                 self.plot_dir + 'opal_plots/')
            s = s + 1
            print('\t <----------------------------------------------------------------------------------->\n')

        print(f_rho.shape, f_kappa.shape, t_f.shape)

    def treat_tasks_tlim(self, n_out, t1, t2, k1 = None, k2 = None, plot = True):

        self.check_t_lim(t1,t2)

        t1, t2, k1, k2, it1, it2 = self.check_lim_task(t1, t2, k1, k2)

        self.min_k = np.array(self.min_k) # for some reason the Error was, that min_k is a list not a np.array
        self.max_k = np.array(self.max_k)

        if Table_Analyze.plot_k_vs_t:
            PhysPlots.k_vs_t(self.t, self.min_k, self.max_k, True, True, k1, k2, t1, t2, it1, it2, self.plot_dir)  # save but not show

        i_1 = [i for i in range(len(self.t)) if self.t[i] == it1][0]        # for self.t as rho2d and kappa2d have indexes as t
        i_2 = [i for i in range(len(self.t)) if self.t[i] == it2][0] + 1    # + 1 added so if t2 = 5.5 it goes up to 5.5.

        print('====================================INTERPOLATING====================================')
        print('\t__Note: Limits for kappa are: ',
              "%.2f" % k1, ' ', "%.2f" % k2, '\n\t  t range is: ', t1, ' ', t2)

        # c = 0  # for cases and appending arrays

        # ii_1 = [i for i in range(len(self.t)) if self.t[i] ==  self.t[0]][0]    # for self.t as rho2d and kappa2d have indexes as t
        # ii_2 = [i for i in range(len(self.t)) if self.t[i] == self.t[-1]][0]

        f_kappa = np.zeros((i_2 - i_1, n_out))  # all rows showld be the same!
        f_rho = np.zeros((i_2 - i_1,   n_out))
        f_t = []

        s = 0
        for i in range(i_1, i_2):
            print('\t <---------------------------| t[', i, '] = ', self.t[i], ' |---------------------------->\n')
            tmp1 = Row_Analyze.clear_row(self.rho2d[i, :], self.kappa2d[i, :],
                                         self.o_cut_mask,
                                         self.o_cut_rep,
                                         self.o_cut_smooth)
            tmp2 = Row_Analyze.solve_for_row(k1, k2, self.cases[i], n_out, tmp1[0], tmp1[1])
            f_rho[s, :]   = np.array(tmp2[0])
            f_kappa[s, :] = np.array(tmp2[1])
            f_t = np.append(f_t, self.t[i])

            if plot:
                print("\t\tPLotting...")
                tmp3 = Row_Analyze.cut_mask_val(self.rho2d[i, :], self.kappa2d[i, :])  # I have to cut masked values, or they
                PhysPlots.Rho_k_plot(tmp2[0], tmp2[1], tmp3[0], tmp3[1],  # screw up the scale of the plot :(
                                     k1, k2, self.cases[i], self.t[i], i, self.plot_dir)
            s = s + 1
            print('\t <----------------------------------------------------------------------------------->\n')
        print('\t__Note: t_limited output: (--) k: {}  (|) t: {}  (|-) rho: {}'.format(f_kappa[0,:].shape, f_t.shape, f_rho.shape))


        return Math.combine(f_kappa[0,:], f_t, f_rho)

    def treat_tasks_interp_for_t(self, t1, t2, n_out, n_interp, k1 = None, k2 = None):

        # self.check_t_lim(t1, t2)

        res = self.treat_tasks_tlim(n_out, t1, t2, k1, k2)
        kap = res[0,1:]
        t   = res[1:,0]
        rho = res[1:,1:]

        print('\t__Note: Performing interpolation from t:', len(t), ' points to', n_interp)


        # print(n_interp, len(kap))
        new_rho = np.zeros((n_interp, len(kap)))
        new_t = np.mgrid[t1: t2: n_interp * 1j]

        # PhysPlots.rho_vs_t(t,rho[:,1])

        for i in range(len(kap)):
            new_rho[:, i] = Math.interp_row(t, rho[:,i], new_t)

        print('\t__Note: t_interp output: (--) k: {}  (|) t: {}  (|-) rho: {}'.format(kap.shape, new_t.shape, new_rho.shape))

        # PhysPlots.rho_vs_t(new_t, new_rho[:, 1]) # one line of rho

        return Math.combine(kap, new_t, new_rho)

    def interp_for_single_k(self, t1, t2, n_interp, k):
        '''
        returns (0--) t , (1--) rho
        :param t1:
        :param t2:
        :param n_interp:
        :param k:
        :return:
        '''
        # if n_out != 1:
        #     sys.exit('\t___Error: Only n_out = 1 can be used for single k (n_out given is {} ).'.format(n_out))
        # n out already set as 1

        res = self.treat_tasks_interp_for_t(t1, t2, 1, n_interp, k, k) # for 1 k interpolation
        t = res[1:,0]
        rho = res[1:,1]

        print('\t__Note: Single t output: (1--) t: {}  (2--) rho: {}'.format(t.shape, rho.shape))
        return np.vstack((t, rho))


class OPAL_Interpol(Read_Table):

    def __init__(self, table_name, n_anal_):

        # super().__init__(table_name) # inheritance, to avoid calling the instance
        # super(table_name, self).__init__()
        Read_Table.__init__(self, table_name)

        self.rho2d = Physics.get_rho(self.r, self.t)
        self.kappa2d = self.kappas
        self.t = self.t
        self.depth = n_anal_



        '''
            Checks if t1,t2 belogs to t, and rho1, rho2 belogs to corresponding
            raws at given t1 and t2, and if length of tables are right
        '''

        # table = np.array((table))

    # @classmethod
    # def from_OPAL_table(cls, table_name, n):
    #     '''
    #
    #     :param table_name: NAME of the OPAL table to read (with root and extension)
    #     :param n_out_: n of kappas in a grid (1 for single kappa output, for eddington opacity)
    #     :param n_anal_: n of points in interpolation the limits of kappa in all temperatures. Around 1000
    #     :return: a class
    #     '''
    #
    #     cl1 = Read_Table(table_name)
    #
    #     r = cl1.r
    #     t = cl1.t
    #     kap = cl1.kappas
    #     rho = Physics.get_rho(r, t)
    #
    #     return cls(rho, kap, t, n)

    def check_t_rho_limits(self, t1, t2,rho1, rho2):

        # Errors.is_a_bigger_b(t1, t2,    '|CheckInputData|', True, 't1 > t2 - must be t1 < t2')
        if t1 > t2:
            raise ValueError('\t__Error! t1({})>t2({}) |OPAL_Interpol|check_t_rho_limits|'.format(t1,t2))

        i = Math.find_nearest_index(self.t, t1)
        j = Math.find_nearest_index(self.t, t2)

        if (rho1 == None):
            rho1 = self.rho2d[j, 1]; print('\t__Note: Smallest rho in the given T range: ', rho1)
        if (rho2 == None):
            rho2 = self.rho2d[i, len(self.rho2d[i, :]) - 1]; print('\t__Note: Largest rho in the given T range ', rho2)

        # Errors.is_a_bigger_b(rho1, rho2,         '|CheckInputData|', True, 'rho1 > rho2, must be rho1 < rho2')
        if rho1 > rho2:
            raise ValueError('\t___Error. rho1({}) > rho2({}) in | OPAL_Interpol | check_t_rho_limits|'.format(rho1,rho2))

        if self.rho2d[j, 0] > rho1:
            raise ValueError('\t__Error:  rho1 ({}) < rho[0] ({}) |check_t_rho_limits|'.format(rho1, self.rho2d[j, 0]))

        if self.rho2d[i, -1] < rho2:
            raise ValueError('\t__Error:  rho2 ({}) > rho[-1] ({}) |check_t_rho_limits|'.format(rho2, self.rho2d[i, -1]))

        print('\t__Note: Overall: min_ro:', self.rho2d.min(),  ' max_rho: ', self.rho2d.max())
        print('\t__Note: Min_ro in T area:', self.rho2d[j, 0], ' max_rho in T area: ', self.rho2d[i, len(self.rho2d[i, :]) - 1])

        return np.array([t1, t2, rho1, rho2])

    # @staticmethod
    # def interpolate_2d(x, y, z, x_coord, y_coord, depth):
    #
    #     x_coord = np.array(x_coord, dtype=float)
    #     y_coord = np.array(y_coord, dtype=float)
    #     # interpolating every row< going down in y.
    #     if len(x_coord)!=len(y_coord):
    #         raise ValueError('x and y coord must be equal in length (x:{}, y:{})'.format(len(x_coord),len(y_coord)))
    #     #
    #     # if x_coord.min() < x.min() or x_coord.max() > x.max():
    #     #     raise ValueError('x_min:{} < x.min:{} or x_max:{} > '.format(x_coord.min(), x.min()))
    #     # if
    #     #
    #     print(x.shape, y.shape, z.shape)
    #     new_z = []#np.zeros((len(y), len(x_coord)))
    #     for si in range(len(y)):
    #         # new_x[si,:] = Math.interp_row(x, z[si,:], x_coord)
    #         new_z = np.append(new_z, Math.interp_row(x, z[si,:], x_coord))
    #
    #     # inteprolating every column, going right in x.
    #     new_z2 = []#np.zeros(( len(y_coord), len(new_x[:,0]) ))
    #     for si in range(len(x)):
    #         # new_y[:,si] = Math.interp_row(y, new_x[:,si], y_coord)
    #         new_z2 = np.append(new_z2, Math.interp_row(y, new_z, y_coord))
    #
    #     print(new_z.shape, new_z2.shape)
    #
    #
    #     f = interpolate.interp2d(x, y, z, kind='cubic')
    #
    #
    #     print(f(x_coord, y_coord))
    #
    #
    #     return None


    def interp_opal_table(self, t1, t2, rho1 = None, rho2 = None):
        '''
            Conducts 2 consequent interplations.
            1st along each raw at const temperature, establishing a grid of new rho
            2nd anlog each line of const rho (from grid) interpolates columns
        '''
        # print('ro1: ',rho1, ' ro2: ',rho2)

        t1, t2, rho1, rho2 = self.check_t_rho_limits(t1,t2,rho1,rho2)


        crop_rho = np.mgrid[rho1:rho2:self.depth * 1j]  # print(crop_rho)
        crop_t = []
        crop_k = np.zeros((len(self.t), self.depth))

        for si in range(len(self.t)):
            if (rho1 > self.rho2d[si, 0] and rho2 < self.rho2d[si, len(self.rho2d[si, :]) - 1]):

                clean_arrays = Row_Analyze.clear_row(self.rho2d[si, :], self.kappa2d[si, :], True, False, False)
                crop_k[si, :] = Math.interp_row(clean_arrays[0, :], clean_arrays[1, :], crop_rho)
                crop_t = np.append(crop_t, self.t[si])

        crop_k = crop_k[~(crop_k == 0).all(1)]  # Removing ALL=0 raws, to leave only filled onese

        extend_k = np.zeros((self.depth, self.depth))
        extend_crop_t = np.mgrid[t1:t2: self.depth * 1j]

        for si in range(self.depth):
            extend_k[:, si] = Math.interp_row(crop_t, crop_k[:, si], extend_crop_t)

        if (len(extend_k[0, :]) != len(crop_rho)): raise ValueError("N of columns in new table not equal to length of rho")
        if (len(extend_k[:, 0]) != len(extend_crop_t)): raise ValueError("N of raws in new table not equal to length of t")

        #    print extend_crop_t.shape, crop_rho.shape, extend_k.shape

        print('\t__Note interp_opal_table out: (--) t: {}  (|) rho: {}  (|-) k: {}'.format(extend_crop_t.shape, crop_rho.shape,
                                                                                      extend_k.T.shape))
        return Math.combine(extend_crop_t, crop_rho, extend_k.T)


class New_Table:
    def __init__(self, path, tables, values, out_dir_name):
        self.plot_dir_name = out_dir_name
        self.tbls = []
        self.val = values
        if len(values)!=len(tables):
            raise ValueError('\t___Error. |New_table, init| n of tables and values is different: {} != {}'.format(len(tables), len(values)))
        self.ntbls = 0
        for i in range(len(tables)):
            self.tbls.append(Read_Table(path + tables[i] + '.data'))
            self.ntbls = self.ntbls + 1


        print('\t__Note: {} opal tables files has been uploaded.'.format(self.ntbls))

    def check_if_opals_same_range(self):
        for i in range(self.ntbls-1):

            if not np.array_equal(self.tbls[i].r, self.tbls[i+1].r):
                raise ValueError('\t___Error. Arrays *r* are not equal '
                         '|check_if_opals_same_range| \n r[{}]: {} \n r[{}]: {}'.format(i, self.tbls[i].r, i+1, self.tbls[i+1].r))

            if not np.array_equal(self.tbls[i].t, self.tbls[i+1].t):
                raise ValueError('\t___Error. Arrays *t* are not equal '
                         '|check_if_opals_same_range| \n t[{}]: {} \n t[{}]: {}'.format(i, self.tbls[i].r, i+1, self.tbls[i+1].r))


        # for i in range(len(self.tbls[0].r)):
        #     pass

    def get_new_opal(self, value, mask = 9.999):
        if value > self.val[-1] or value < self.val[0]:
            raise ValueError('\t___Error. |get_new_opal| value {} is not is range of tables: ({}, {})'.format(value, self.val[0],self.val[-1]))

        self.check_if_opals_same_range()

        rows = len(self.tbls[0].t)
        cols = len(self.tbls[0].r)

        new_kappa = np.zeros((rows, cols))
        for i in range(rows):
            for j in range(cols):

                val_aval = []
                k_row = []
                for k in range(self.ntbls):
                    k_val = self.tbls[k].kappas[i,j]
                    if k_val != mask:
                        k_row = np.append(k_row, self.tbls[k].kappas[i,j])
                        val_aval = np.append(val_aval, self.val[k])

                if len(val_aval) == 0:
                    new_kappa[i, j] = mask
                else:
                    if value >= val_aval[0] and value <= val_aval[-1]:
                        new_kappa[i, j] = Math.interp_row(val_aval, k_row, value)
                    else:
                        new_kappa[i, j] = mask

        # "%.2f" %
        fname = self.plot_dir_name + 'table_x.data'
        res = Math.combine(self.tbls[0].r,self.tbls[0].t, new_kappa)
        np.savetxt(fname,res,'%.3f','\t')
        return res


class PlotBackground:


    def __init__(self):
        pass

    @staticmethod
    def plot_color_table(table, v_n_x, v_n_y, v_n_z, opal_used, label = None, fsz=12, lagel_angle=0):

        plt.figure()
        ax = plt.subplot(111)


        if label != None:
            ax.text(0.8, 0.1, label, style='italic',
                    bbox={'facecolor': 'yellow', 'alpha': 0.5, 'pad': 10}, horizontalalignment='center',
                    verticalalignment='center', transform=ax.transAxes)


            # print('TEXT')
            # plt.text(table[0, 1:].min(), table[1:, 0].min(), label, style='italic')
            # bbox={'facecolor': 'white', 'alpha': 0.5, 'pad': 10}
            # plt.text(2, 0.65, r'$\cos(2 \pi t) \exp(-t)$')

        # ax = fig.add_subplot(1, 1, 1)
        ax.set_xlim(table[0, 1:].min(), table[0, 1:].max())
        ax.set_ylim(table[1:, 0].min(), table[1:, 0].max())
        ax.set_ylabel(Labels.lbls(v_n_y), fontsize=fsz)
        ax.set_xlabel(Labels.lbls(v_n_x), fontsize=fsz)

        levels = Levels.get_levels(v_n_z, opal_used)

        contour_filled = plt.contourf(table[0, 1:], table[1:, 0], table[1:, 1:], levels, cmap=plt.get_cmap('RdYlBu_r'),
                                      alpha=1.0)
        clb = plt.colorbar(contour_filled) # orientation='horizontal', :)
        clb.ax.set_title(Labels.lbls(v_n_z), fontsize=fsz)
        clb.ax.tick_params(labelsize=fsz)

        # ax.colorbar(contour_filled, label=Labels.lbls(v_n_z))
        contour = plt.contour(table[0, 1:], table[1:, 0], table[1:, 1:], levels, colors='k')
        labs = ax.clabel(contour, colors='k', fmt='%2.2f', inline=True, fontsize=fsz)
        for lab in labs:
            lab.set_rotation(lagel_angle)#295
        # contour_filled = plt.contourf(table[0, 1:], table[1:, 0], table[1:, 1:], levels, cmap=plt.get_cmap('RdYlBu_r'))
        # clb = plt.colorbar(contour_filled)
        # clb.ax.set_title(Labels.lbls(v_n_z), fontsize=fsz)
        # contour = plt.contour(table[0, 1:], table[1:, 0], table[1:, 1:], levels, colors='k')
        # # plt.clabel(contour, colors='k', fmt='%2.2f', fontsize=fsz)
        # # plt.title('SONIC HR DIAGRAM')

        ax.minorticks_on()

        ax.invert_xaxis()

        plt.xticks(fontsize=fsz)
        plt.yticks(fontsize=fsz)

        # plt.ylabel(l_or_lm)
        # plt.legend(bbox_to_anchor=(0, 0), loc='lower left', ncol=1)
        # plt.savefig(name)
        plt.show()

    @staticmethod
    def plot_color_background(ax, table, v_n_x, v_n_y, v_n_z, opal_used, bump, label = None, alpha = 0.8, clean=False, fsz=12, rotation=0):



        # if label != None:
        #     print('TEXT')

            # ax.text(table[0, 1:].min(), table[1:, 0].min(), s=label)
            # bbox={'facecolor': 'white', 'alpha': 0.5, 'pad': 10}
            # plt.text(2, 0.65, r'$\cos(2 \pi t) \exp(-t)$')

        # ax = fig.add_subplot(1, 1, 1)

        ax.set_xlim(table[0, 1:].min(), table[0, 1:].max())
        ax.set_ylim(table[1:, 0].min(), table[1:, 0].max())
        ax.set_ylabel(Labels.lbls(v_n_y), fontsize=fsz)
        ax.set_xlabel(Labels.lbls(v_n_x), fontsize=fsz)

        levels = Levels.get_levels(v_n_z, opal_used, bump)

        # 'RdYlBu_r'

        if v_n_x == 'mdot' and v_n_y == 'lm' and v_n_z == 'tau':
            pass
        else:
            contour_filled = plt.contourf(table[0, 1:], table[1:, 0], table[1:, 1:], levels, cmap=plt.get_cmap('RdYlBu_r'), alpha=alpha)
            clb = plt.colorbar(contour_filled)
            clb.ax.set_title(Labels.lbls(v_n_z), fontsize=fsz)
            clb.ax.tick_params(labelsize=fsz)

        # ax.colorbar(contour_filled, label=Labels.lbls(v_n_z))

        contour = plt.contour(table[0, 1:], table[1:, 0], table[1:, 1:], levels, colors='k')

        if v_n_x == 'mdot' and v_n_y == 'lm' and v_n_z == 'tau':
            labs=ax.clabel(contour, colors='k', fmt='%2.0f', fontsize=fsz, manual=True)
        else:
            labs = ax.clabel(contour, colors='k', fmt='%2.2f', fontsize=fsz)

        if rotation != None:
            for lab in labs:
                lab.set_rotation(rotation)       # ORIENTATION OF LABELS IN COUNTUR PLOTS
        # ax.set_title('SONIC HR DIAGRAM')

        # print('Yc:{}'.format(yc_val))
        if not clean and label != None and label != '':
            ax.text(0.9, 0.1, label, style='italic',
                    bbox={'facecolor': 'yellow', 'alpha': 0.5, 'pad': 10}, horizontalalignment='center',
                    verticalalignment='center', transform=ax.transAxes)

        ax.tick_params('y', labelsize=fsz)
        ax.tick_params('x', labelsize=fsz)

        # plt.ylabel(l_or_lm)
        # plt.legend(bbox_to_anchor=(0, 0), loc='lower left', ncol=1)
        # plt.savefig(name)
        # plt.show()
        return ax

    # @staticmethod
    # def plot_obs_x_llm(ax, obs_cls, l_or_lm, v_n_x, yc_val, use_gaia = False, clean=False, check_star_lm_wne=False):
    #
    #     def plot_obs_err(ax, obs_cls, star_n, l_or_lm, v_n_x, yc_val, use_gaia = False):
    #
    #         x_coord =  obs_cls.get_num_par(v_n_x, star_n)
    #
    #         if l_or_lm == 'lm':
    #
    #             lm_err1, lm_err2 = obs_cls.get_star_lm_obs_err(star_n, yc_val, use_gaia)          # ERRORS Mdot
    #
    #             if v_n_x == 'mdot':
    #                 mdot1, mdot2 = obs_cls.get_star_mdot_obs_err(star_n, yc_val)
    #                 mdot_coord = [mdot1, mdot2, mdot2, mdot1]
    #                 lm_coord = [lm_err1, lm_err1, lm_err2, lm_err2]
    #                 ax.add_patch(patches.Polygon(xy=list(zip(mdot_coord, lm_coord)), fill=True, alpha=.4,
    #                                              color=obs_cls.get_class_color(star_n)))
    #             else:
    #                 ax.plot([x_coord, x_coord], [lm_err1, lm_err2], '-', color=obs_cls.get_class_color(star_n))
    #
    #         if l_or_lm == 'l':
    #
    #             l_err1, l_err2 = obs_cls.get_star_l_obs_err(star_n, yc_val, use_gaia)
    #
    #             if v_n_x == 'mdot':
    #                 mdot1, mdot2 = obs_cls.get_star_mdot_obs_err(star_n, yc_val)
    #                 mdot_coord = [mdot1, mdot2, mdot2, mdot1]
    #                 l_coord = [l_err1, l_err1, l_err2, l_err2]
    #                 ax.add_patch(patches.Polygon(xy=list(zip(mdot_coord, l_coord)), fill=True, alpha=.4,
    #                                              color=obs_cls.get_class_color(star_n)))
    #             else:
    #                 ax.plot([x_coord, x_coord], [l_err1, l_err2], '-', color=obs_cls.get_class_color(star_n))
    #
    #     def plot_stars(ax, obs_cls, star_n, l_or_lm, v_n_x, yc_val, use_gaia = False, clean=False):
    #
    #         x_coord = obs_cls.get_num_par(v_n_x, star_n)
    #         llm_obs = obs_cls.get_llm(l_or_lm, star_n, yc_val, use_gaia, check_star_lm_wne)
    #
    #         ax.plot(x_coord, llm_obs, marker=obs_cls.get_clss_marker(star_n), markersize='9',
    #                 color=obs_cls.get_class_color(star_n), ls='', mec='black')  # plot color dots)))
    #
    #         if not clean:
    #             ax.annotate('{}'.format(int(star_n)), xy=(x_coord, llm_obs),
    #                         textcoords='data')  # plot numbers of stars
    #
    #     def plot_evol_err(ax, obs_cls, star_n, l_or_lm, v_n_x, yc_val):
    #         if l_or_lm == 'lm':
    #             llm1, llm2 = obs_cls.get_star_lm_err(star_n, yc_val)
    #             # obs_cls.get_star_llm_evol_err(star_n, l_or_lm, yc_val, 1.0, 0.1)                  # ERRORS L/LM
    #             # mdot1, mdot2 = obs_cls.get_star_mdot_err(star_n, l_or_lm, yc_val, 1.0, 0.1, 'nugis')           # ERRORS Mdot
    #
    #             x_coord = obs_cls.get_num_par(v_n_x, star_n)
    #
    #             ax.plot([x_coord, x_coord], [llm1, llm2], '-', color='gray')
    #             ax.plot([x_coord, x_coord], [llm1, llm2], '.', color='gray')
    #
    #
    #
    #     classes = []
    #     classes.append('dum')
    #     x_coord = []
    #     llm_obs = []
    #
    #     # from Phys_Math_Labels import Opt_Depth_Analythis
    #     # use_gaia = False
    #     for star_n in obs_cls.stars_n:
    #         i = -1
    #         x_coord = np.append(x_coord, obs_cls.get_num_par(v_n_x, star_n))
    #         llm_obs = np.append(llm_obs, obs_cls.get_llm(l_or_lm, star_n, yc_val, use_gaia, check_star_lm_wne))
    #         # llm_obs = np.append(llm_obs, obs_cls.get_num_par(l_or_lm, star_n, yc_val, use_gaia))
    #
    #         plot_obs_err(ax, obs_cls, star_n, l_or_lm, v_n_x, yc_val, use_gaia)
    #         plot_stars(ax, obs_cls, star_n, l_or_lm, v_n_x, yc_val, use_gaia, clean)
    #         plot_evol_err(ax,obs_cls, star_n, l_or_lm, v_n_x, yc_val)
    #
    #         if obs_cls.get_star_class(star_n) not in classes:
    #             plt.plot(x_coord[i], llm_obs[i], marker=obs_cls.get_clss_marker(star_n), markersize='9',
    #                      color=obs_cls.get_class_color(star_n), ls='', mec='black',
    #                      label='{}'.format(obs_cls.get_star_class(star_n)))  # plot color dots)))
    #             classes.append(obs_cls.get_star_class(star_n))
    #
    #         # # --- PLOT OBSERVABLE ERROR ---
    #         # if l_or_lm == 'lm':
    #         #     llm1, llm2 = obs_cls.get_star_lm_err(star_n, yc_val, use_gaia)          # ERRORS Mdot
    #         #     lm = obs_cls.get_num_par(v_n_x, star_n)
    #         #
    #         #     if v_n_x == 'mdot':
    #         #         mdot1, mdot2 = obs_cls.get_star_mdot_obs_err(star_n, yc_val)
    #         #         mdot_coord = [mdot1, mdot2, mdot2, mdot1]
    #         #         lm_coord = [lm_err1, lm_err1, lm_err2, lm_err2]
    #         #         ax.add_patch(patches.Polygon(xy=list(zip(mdot_coord, lm_coord)), fill=True, alpha=.4,
    #         #                                      color=obs_cls.get_class_color(star_n)))
    #         #     else:
    #         #         ax.plot([x_coord_, x_coord_], [llm1, llm2], '-', color='gray')
    #         #
    #         # else:
    #         #     l1, l2 = obs_cls.get_star_l_err(star_n, yc_val, use_gaia)
    #         #     x_coord_ = obs_cls.get_num_par(v_n_x, star_n)
    #         #     ax.plot([x_coord_, x_coord_], [l1, l2], '-', color='gray')
    #         #
    #         # if v_n_x == 'mdot':
    #         #     mdot1, mdot2 = obs_cls.get_star_mdot_obs_err(star_n, yc_val)
    #         #     ax.plot([x_coord_, x_coord_], [l1, l2], '-', color='gray')
    #
    #             # ax.errorbar(mdot_obs[i], llm_obs[i], yerr=[[llm1],  [llm2]], fmt='--.', color=obs_cls.get_class_color(star_n))
    #
    #         # ax.plot(x_coord[i], llm_obs[i], marker=obs_cls.get_clss_marker(star_n), markersize='9',
    #         #         color=obs_cls.get_class_color(star_n), ls='', mec='black')  # plot color dots)))
    #         # if not clean:
    #         #     ax.annotate('{}'.format(int(star_n)), xy=(x_coord[i], llm_obs[i]),
    #         #                 textcoords='data')  # plot numbers of stars
    #
    #         # t = obs_cls.get_num_par('t', star_n)
    #         # ax.annotate('{}'.format("%.2f" % t), xy=(mdot_obs[i], llm_obs[i]), textcoords='data')  # plot numbers of stars
    #
    #         # v_inf = obs_cls.get_num_par('v_inf', star_n)
    #         # tau_cl = Opt_Depth_Analythis(30, v_inf, 1., 1., mdot_obs[i], 0.20)
    #         # tau = tau_cl.anal_eq_b1(1.)
    #         # # # # ax.annotate(str(int(tau)), xy=(mdot_obs[i], llm_obs[i]), textcoords='data')  # plo
    #         # ax.annotate('{} {}'.format(str(int(tau)), eta), xy=(mdot_obs[i], llm_obs[i]),
    #         #             textcoords='data')  # plot numbers of stars
    #
    #         # if obs_cls.get_star_class(star_n) not in classes:
    #         #     plt.plot(x_coord[i], llm_obs[i], marker=obs_cls.get_clss_marker(star_n), markersize='9',
    #         #              color=obs_cls.get_class_color(star_n), ls='', mec='black',
    #         #              label='{}'.format(obs_cls.get_star_class(star_n)))  # plot color dots)))
    #         #     classes.append(obs_cls.get_star_class(star_n))
    #
    #
    #     print('\t__PLOT: total stars: {}'.format(len(obs_cls.stars_n)))
    #     print(len(x_coord), len(llm_obs))
    #
    #     # fit = np.polyfit(mdot_obs, llm_obs, 1)  # fit = set of coeddicients (highest first)
    #     # f = np.poly1d(fit)
    #     # fit_x_coord = np.mgrid[(mdot_obs.min() - 1):(mdot_obs.max() + 1):1000j]
    #
    #     mdot_grid = np.mgrid[(x_coord.min() - 1):(x_coord.max() + 1):100j]
    #     x_coord__, y_coord__ = Math.fit_plynomial(x_coord, llm_obs, 1, 100, mdot_grid)
    #     ax.plot(x_coord__, y_coord__, '-.', color='blue')
    #
    #     min_mdot, max_mdot = obs_cls.get_min_max('mdot')
    #     min_llm, max_llm = obs_cls.get_min_max_llm(l_or_lm, yc_val, use_gaia, check_star_lm_wne)
    #
    #     # ax.set_xlim(min_mdot - 0.2, max_mdot + 0.2)
    #     # ax.set_ylim(min_llm - 0.05, max_llm + 0.05)
    #
    #     ax.set_ylabel(Labels.lbls(l_or_lm))
    #     ax.set_xlabel(Labels.lbls('mdot'))
    #     ax.grid(which='major', alpha=0.2)
    #     ax.legend(bbox_to_anchor=(1, 1), loc='upper right', ncol=1)
    #
    #     print('Yc:{}'.format(yc_val))
    #
    #     if not clean:
    #         ax.text(0.9, 0.9, 'Yc:{}'.format(yc_val), style='italic',
    #                 bbox={'facecolor': 'yellow', 'alpha': 0.5, 'pad': 10}, horizontalalignment='center',
    #                 verticalalignment='center', transform=ax.transAxes)
    #         if use_gaia:
    #             ax.text(0.9, 0.75, 'GAIA', style='italic',
    #                     bbox={'facecolor': 'blue', 'alpha': 0.5, 'pad': 10}, horizontalalignment='center',
    #                     verticalalignment='center', transform=ax.transAxes)
    #
    #     return ax
    #
    # @staticmethod
    # def plot_obs_mdot_llm(ax, obs_cls, l_or_lm, yc_val, clean = False):
    #     '''
    #
    #     :param ax:
    #     :param obs_cls:
    #     :param l_or_lm:
    #     :param yc_val:
    #     :param yc1:
    #     :param yc2:
    #     :return:
    #     '''
    #     classes = []
    #     classes.append('dum')
    #     mdot_obs = []
    #     llm_obs = []
    #
    #     # from Phys_Math_Labels import Opt_Depth_Analythis
    #
    #     for star_n in obs_cls.stars_n:
    #         i = -1
    #         mdot_obs = np.append(mdot_obs, obs_cls.get_num_par('mdot', star_n))
    #         llm_obs = np.append(llm_obs, obs_cls.get_num_par(l_or_lm, star_n, yc_val))
    #         eta = obs_cls.get_num_par('eta', star_n)
    #
    #         lm_err1, lm_err2 = obs_cls.get_star_lm_obs_err(star_n, yc_val)
    #         mdot1, mdot2 = obs_cls.get_star_mdot_obs_err(star_n, yc_val)
    #
    #         mdot_coord = [mdot1, mdot2, mdot2, mdot1]
    #         lm_coord = [lm_err1, lm_err1, lm_err2, lm_err2]
    #         ax.add_patch(patches.Polygon(xy=list(zip(mdot_coord, lm_coord)), fill=True, alpha=.4,
    #                                      color=obs_cls.get_class_color(star_n)))
    #
    #         if l_or_lm == 'lm':
    #
    #             llm1, llm2 = obs_cls.get_star_lm_err(star_n, yc_val)
    #                 # obs_cls.get_star_llm_evol_err(star_n, l_or_lm, yc_val, 1.0, 0.1)                  # ERRORS L/LM
    #             # mdot1, mdot2 = obs_cls.get_star_mdot_err(star_n, l_or_lm, yc_val, 1.0, 0.1, 'nugis')           # ERRORS Mdot
    #             mdot = obs_cls.get_num_par('mdot', star_n)
    #             ax.plot([mdot, mdot], [llm1, llm2], '-', color='white')
    #             #color=obs_cls.get_class_color(star_n)
    #
    #             # ax.errorbar(mdot_obs[i], llm_obs[i], yerr=[[llm1],  [llm2]], fmt='--.', color=obs_cls.get_class_color(star_n))
    #
    #
    #         ax.plot(mdot_obs[i], llm_obs[i], marker=obs_cls.get_clss_marker(star_n), markersize='9',
    #                  color=obs_cls.get_class_color(star_n), ls='', mec='black')  # plot color dots)))
    #         if not clean:
    #             ax.annotate('{}'.format(int(star_n)), xy=(mdot_obs[i], llm_obs[i]),textcoords='data')  # plot numbers of stars
    #
    #         # t = obs_cls.get_num_par('t', star_n)
    #         # ax.annotate('{}'.format("%.2f" % t), xy=(mdot_obs[i], llm_obs[i]), textcoords='data')  # plot numbers of stars
    #
    #
    #         # v_inf = obs_cls.get_num_par('v_inf', star_n)
    #         # tau_cl = Opt_Depth_Analythis(30, v_inf, 1., 1., mdot_obs[i], 0.20)
    #         # tau = tau_cl.anal_eq_b1(1.)
    #         # # # # ax.annotate(str(int(tau)), xy=(mdot_obs[i], llm_obs[i]), textcoords='data')  # plo
    #         # ax.annotate('{} {}'.format(str(int(tau)), eta), xy=(mdot_obs[i], llm_obs[i]),
    #         #             textcoords='data')  # plot numbers of stars
    #
    #         if obs_cls.get_star_class(star_n) not in classes:
    #             plt.plot(mdot_obs[i], llm_obs[i], marker=obs_cls.get_clss_marker(star_n), markersize='9',
    #                      color=obs_cls.get_class_color(star_n), ls='', mec='black',
    #                      label='{}'.format(obs_cls.get_star_class(star_n)))  # plot color dots)))
    #             classes.append(obs_cls.get_star_class(star_n))
    #
    #     print('\t__PLOT: total stars: {}'.format(len(obs_cls.stars_n)))
    #     print(len(mdot_obs), len(llm_obs))
    #
    #     # fit = np.polyfit(mdot_obs, llm_obs, 1)  # fit = set of coeddicients (highest first)
    #     # f = np.poly1d(fit)
    #     # fit_x_coord = np.mgrid[(mdot_obs.min() - 1):(mdot_obs.max() + 1):1000j]
    #
    #     mdot_grid = np.mgrid[(mdot_obs.min() - 1):(mdot_obs.max() + 1):100j]
    #     x_coord, y_coord = Math.fit_plynomial(mdot_obs, llm_obs, 1, 100, mdot_grid)
    #     ax.plot(x_coord, y_coord, '-.', color='blue')
    #
    #     min_mdot, max_mdot = obs_cls.get_min_max('mdot')
    #     min_llm, max_llm = obs_cls.get_min_max(l_or_lm, yc_val)
    #
    #     ax.set_xlim(min_mdot - 0.2, max_mdot + 0.2)
    #     ax.set_ylim(min_llm - 0.05, max_llm + 0.05)
    #
    #     ax.set_ylabel(Labels.lbls(l_or_lm))
    #     ax.set_xlabel(Labels.lbls('mdot'))
    #     ax.grid(which='major', alpha=0.2)
    #     ax.legend(bbox_to_anchor=(1, 1), loc='upper right', ncol=1)
    #
    #     print('Yc:{}'.format(yc_val))
    #
    #     if not clean:
    #         ax.text(0.9, 0.9, 'Yc:{}'.format(yc_val), style='italic',
    #                 bbox={'facecolor': 'yellow', 'alpha': 0.5, 'pad': 10}, horizontalalignment='center',
    #                 verticalalignment='center', transform=ax.transAxes)
    #
    #     return ax
    #     # ax.text(min_mdot, max_llm, 'Yc:{}'.format(yc_val), style='italic',
    #     #         bbox={'facecolor': 'yellow', 'alpha': 0.5, 'pad': 10})
    #
    #     # l_grid = np.mgrid[5.2:6.:100j]
    #     # ax.plot(Physics.l_mdot_prescriptions(l_grid, ), l_grid, '-.', color='orange', label='Nugis & Lamers 2000')
    #     #
    #     # ax.plot(Physics.yoon(l_grid, 10 ** 0.02), l_grid, '-.', color='green', label='Yoon 2017')
    #
    # @staticmethod
    # def plot_obs_t_llm_mdot_int(ax, t_llm_mdot, obs_cls, l_or_lm, lim_t1 = None, lim_t2 = None, show_legend = True, clean = False):
    #
    #     if lim_t1 == None: lim_t1 = t_llm_mdot[0, 1:].min()
    #     if lim_t2 == None: lim_t2 = t_llm_mdot[0, 1:].max()
    #
    #     yc_val = t_llm_mdot[0, 0] #
    #
    #     classes = []
    #     classes.append('dum')
    #     x = []
    #     y = []
    #     for star_n in obs_cls.stars_n:
    #         xyz = obs_cls.get_xyz_from_yz(yc_val, star_n, l_or_lm, 'mdot',
    #                                       t_llm_mdot[0,1:], t_llm_mdot[1:,0], t_llm_mdot[1:,1:], lim_t1, lim_t2)
    #
    #         if xyz.any():
    #             x = np.append(x, xyz[0, 0])
    #             y = np.append(y, xyz[1, 0])
    #
    #             # print('Star {}, {} range: ({}, {})'.format(star_n,l_or_lm, llm1, llm2))
    #
    #             for i in range(len(xyz[0, :])):
    #
    #                 ax.plot(xyz[0, i], xyz[1, i], marker=obs_cls.get_clss_marker(star_n), markersize='9',
    #                          color=obs_cls.get_class_color(star_n), ls='', mec='black')  # plot color dots)))
    #                 if not clean:
    #                     ax.annotate(int(star_n), xy=(xyz[0, i], xyz[1, i]),
    #                                 textcoords='data')  # plot numbers of stars
    #
    #                 if obs_cls.get_star_class(star_n) not in classes:
    #                     ax.plot(xyz[0, i], xyz[1, i], marker=obs_cls.get_clss_marker(star_n), markersize='9',
    #                              color=obs_cls.get_class_color(star_n), mec='black', ls='',
    #                              label='{}'.format(obs_cls.get_star_class(star_n)))  # plot color dots)))
    #                     classes.append(obs_cls.get_star_class(star_n))
    #
    #                 # -------------------------OBSERVABLE ERRORS FOR L and Mdot ----------------------------------------
    #                 lm_err1, lm_err2 = obs_cls.get_star_lm_obs_err(star_n, yc_val)
    #                 ts1_b, ts2_b, ts1_t, ts2_t = obs_cls.get_star_ts_obs_err(star_n, t_llm_mdot, yc_val, lim_t1, lim_t2)
    #                 ts_coord = [ts1_b, ts2_b, ts2_t, ts1_t]
    #                 lm_coord = [lm_err1, lm_err1, lm_err2, lm_err2]
    #                 ax.add_patch(patches.Polygon(xy=list(zip(ts_coord, lm_coord)), fill=True, alpha=.7,
    #                                              color=obs_cls.get_class_color(star_n)))
    #
    #                 # ax.plot([xyz[0, i], xyz[0, i]], [lm_err1, lm_err2], '-',
    #                 #         color='gray')
    #
    #                 # ax.plot([ts_err1, ts_err2], [xyz[1, i], xyz[1, i]], '-',
    #                 #         color='gray')
    #                 # print('  :: Star {}. L/M:{}(+{}/-{}) ts:{}(+{}/-{})'
    #                 #       .format(star_n, "%.2f" % xyz[0, i], "%.2f" % np.abs(xyz[0, i]-lm_err2), "%.2f" % np.abs(xyz[0, i]-lm_err2),
    #                 #               "%.2f" % xyz[1, i],  "%.2f" % np.abs(xyz[1, i]-ts_err2), "%.2f" % np.abs(xyz[1, i]-ts_err1)))
    #                 # ax.plot([ts_err1, ts_err2], [lm_err1, lm_err2], '-', color='gray')
    #
    #                 #
    #
    #
    #
    #                 if l_or_lm == 'lm':
    #                     lm1, lm2 = obs_cls.get_star_lm_err(star_n, yc_val)
    #                     ts1, ts2 = obs_cls.get_star_ts_err(star_n, t_llm_mdot, yc_val, lim_t1, lim_t2)
    #                     ax.plot([ts1, ts2], [lm1, lm2], '-', color='white')
    #                             # color=obs_cls.get_class_color(star_n))
    #
    #                     # ax.plot([xyz[0, i], xyz[0, i]], [lm1, lm2], '-',
    #                     #         color=obs_cls.get_class_color(star_n))
    #                     # ax.plot([ts1, ts2], [xyz[1, i], xyz[1, i]], '-',
    #                     #         color=obs_cls.get_class_color(star_n))
    #
    #
    #                     # ax.add_patch(patches.Rectangle((xyz[0, i] - ts1, xyz[1, i] - lm1), ts2 + ts1, lm2 + lm1,
    #                     #                                alpha=.3, color=obs_cls.get_class_color(star_n)))
    #
    #                     # ax.add_patch(patches.Rectangle((xyz[0, i] - ts1, xyz[1, i] - lm1), ts2 + ts1, lm2 + lm1,
    #                     #                                alpha=.3, color=obs_cls.get_class_color(star_n)))
    #
    #                     # ax.plot([xyz[0, i] - ts1, xyz[1, i] - lm1], [xyz[0, i]+ts2, xyz[1, i] + lm2], '-', color=obs_cls.get_class_color(star_n))
    #                     # ax.plot([xyz[0, i] - ts1, xyz[0, i]+ts2], [xyz[1, i] - lm1, xyz[1, i] + lm2], '-', color=obs_cls.get_class_color(star_n))
    #
    #                     # ax.errorbar(xyz[0, i], xyz[1, i], yerr=[[lm1], [lm2]], fmt='--.', color = obs_cls.get_class_color(star_n))
    #                     # ax.errorbar(xyz[0, i], xyz[1, i], xerr=[[ts1], [ts2]], fmt='--.', color=obs_cls.get_class_color(star_n))
    #
    #
    #
    #     fit = np.polyfit(x, y, 1)  # fit = set of coeddicients (highest first)
    #     f = np.poly1d(fit)
    #     fit_x_coord = np.mgrid[(t_llm_mdot[0,1:].min()):(t_llm_mdot[0,1:].max()):1000j]
    #     ax.plot(fit_x_coord, f(fit_x_coord), '-.', color='blue')
    #
    #     ax.set_xlim(t_llm_mdot[0,1:].min(), t_llm_mdot[0,1:].max())
    #     ax.set_ylim(t_llm_mdot[1:,0].min(), t_llm_mdot[1:,0].max())
    #     # ax.text(0.9, 0.9,'Yc:{}'.format(yc_val), style='italic',
    #     #         bbox={'facecolor': 'yellow', 'alpha': 0.5, 'pad': 10}, horizontalalignment='center', verticalalignment='center', transform = ax.transAxes)
    #
    #     # ax.text(x.max(), y.max(), 'Yc:{}'.format(yc_val), style='italic',
    #     #         bbox={'facecolor': 'yellow', 'alpha': 0.5, 'pad': 10})
    #     if show_legend:
    #         ax.legend(bbox_to_anchor=(0, 0), loc='lower left', ncol=1)
    #
    #     return ax



""" ------------------- """


def metalicity(metal):
    if metal.split('/')[-1] == 'gal':
        return 0.02
    if metal.split('/')[-1] == 'lmc':
        return 0.008
    if metal.split('/')[-1] == '2gal':
        return 0.04
    if metal.split('/')[-1] == 'smc':
        return 0.004
    raise NameError('Metallicity is not recognised: (given: {})'.format(metal))

def get_opal(gal_or_lmc):
    if gal_or_lmc == 'gal': return '../data/opal/table8.data'
    if gal_or_lmc == 'lmc': return '../data/opal/table_x.data'
    if gal_or_lmc == '2gal': return '../data/opal/table10.data'
    if gal_or_lmc == 'smc': return '../data/opal/table6.data'

    raise NameError('Only gal_or_lmc avaialbale for OPAL files, given: {},'
                    .format(gal_or_lmc))

def t_for_bump(bump_name):
    '''
    Returns t1, t2 for the opacity bump, corresponding to a Fe or HeII increase
    :param bump_name:
    :return:
    '''

    t_fe_bump1 = 5.19  # 5.21 # 5.18
    t_fe_bump2 = 5.45

    t_he2_bump1 = 4.65
    t_he2_bump2 = 5.00

    if bump_name == 'HeII':
        return t_he2_bump1, t_he2_bump2

    if bump_name == 'Fe':
        return t_fe_bump1, t_fe_bump2

    raise NameError('Incorrect bump_name. Opacity bumps availabel: {}'.format(['HeII', 'Fe']))

class OPAL_work:

    def __init__(self, metal, bump, n_interp = 1000, load_lim_cases = False,
                 output_dir = './data/output/', plot_dir = './plots/'):

        self.set_metal = metal
        self.op_name = get_opal(metal)
        self.bump = bump
        self.t1, self.t2 = t_for_bump(bump)
        self.n_inter = n_interp
        self.set_plots_clean = False

        self.out_dir = output_dir
        self.plot_dir = plot_dir

        self.opal = OPAL_Interpol(get_opal(metal), n_interp)
        self.tbl_anl = Table_Analyze(get_opal(metal), n_interp, load_lim_cases, output_dir, plot_dir)

    def save_t_rho_k(self, rho1 = None, rho2=None, t1=None, t2=None):
        if t1==None: t1 = self.t1
        if t2==None: t2 = self.t2

        op_cl = self.opal
        t1, t2, rho1, rho2 = op_cl.check_t_rho_limits(t1, t2, rho1, rho2)
        op_table = op_cl.interp_opal_table(t1, t2, rho1, rho2)

        Save_Load_tables.save_table(op_table, self.set_metal, self.bump, 't_rho_k', 't', 'rho', 'k', self.out_dir)

    def save_t_k_rho(self, llm1=None, llm2=None, n_out = 1000):

        k1, k2 = Physics.get_k1_k2_from_llm1_llm2(self.t1, self.t2, llm1, llm2) # assuming k = 4 pi c G (L/M)

        global t_k_rho
        t_k_rho = self.tbl_anl.treat_tasks_interp_for_t(self.t1, self.t2, n_out, self.n_inter, k1, k2).T

        t_k_rho__ = Math.combine(t_k_rho[0,1:], 10**t_k_rho[1:,0], t_k_rho[1:,1:])

        lbl = 'z:{}'.format(metalicity(self.set_metal))
        if self.set_plots_clean: lbl = None

        PlotBackground.plot_color_table(t_k_rho__, 't', 'k', 'rho', self.set_metal, lbl)

        Save_Load_tables.save_table(t_k_rho, self.set_metal, self.bump, 't_k_rho', 't', 'k', 'rho', self.out_dir)
        print('\t__Note. Table | t_k_rho | has been saved in {}'.format(self.out_dir))
        # self.read_table('t_k_rho', 't', 'k', 'rho', self.op_name)
        # def save_t_llm_vrho(self, llm1=None, llm2=None, n_out = 1000):

    def from_t_k_rho__to__t_lm_rho(self, coeff = 1.0):
        t_k_rho = Save_Load_tables.load_table('t_k_rho', 't', 'k', 'rho', self.set_metal, self.bump)

        t = t_k_rho[0, 1:]
        k = t_k_rho[1:, 0]
        rho2d = t_k_rho[1:, 1:]

        lm = Physics.logk_loglm(k, 1, coeff)

        t_lm_rho = Math.invet_to_ascending_xy( Math.combine(t, lm, rho2d) )

        Save_Load_tables.save_table(t_lm_rho, self.set_metal, self.bump, 't_{}lm_rho'.format(coeff), 't', '{}lm'.format(coeff), 'rho', self.out_dir)
        print('\t__Note. Table | t_lm_rho | for | {} | has been saved in {}'.format(self.op_name, self.out_dir))

    def save_t_llm_vrho(self, l_or_lm_name):
        '''
        Table required: t_k_rho (otherwise - won't work) [Run save_t_k_rho() function ]
        :param l_or_lm_name:
        :return:
        '''

        # 1 load the t_k_rho
        t_k_rho = Save_Load_tables.load_table('t_k_rho', 't', 'k', 'rho', self.set_metal, self.bump)

        k = t_k_rho[0, 1:]
        t = t_k_rho[1:, 0]
        rho2d = t_k_rho[1:, 1:]

        vrho = Physics.get_vrho(t, rho2d.T, 2) # mu = 1.34 by default | rho2d.T is because in OPAL t is Y axis, not X.

        # ----------------------------- SELECT THE Y AXIS -----------------
        if l_or_lm_name == 'l':
            l_lm_arr = Physics.lm_to_l_langer(Physics.logk_loglm(k, True))  # Kappa -> L/M -> L
        else:
            l_lm_arr = Physics.logk_loglm(k, 1)


        l_lm_arr = np.flip(l_lm_arr, 0)  # accounting for if k1 > k2 the l1 < l2 or lm1 < lm2
        vrho     = np.flip(vrho, 0)

        global t_llm_vrho
        t_llm_vrho = Math.combine(t, l_lm_arr, vrho)
        name = 't_'+ l_or_lm_name + '_vrho'

        Save_Load_tables.save_table(t_llm_vrho, self.set_metal, self.bump, name, 't', l_or_lm_name, '_vrho', self.out_dir)

        return t_llm_vrho

        # print(t_llm_vrho.shape)

    def save_t_llm_mdot(self, r_s, l_or_lm, r_s_for_t_l_vrho): # mu assumed constant
        '''
        Table required: l_or_lm (otherwise - won't work) [Run save_t_llm_vrho() function ]

        :param r_s: float, 1darray or 2darray
        :param l_or_lm:
        :param r_s_for_t_l_vrho: 't' - means change rows   of   vrho to get mdot (rho = f(ts))
                                 'l' - means change columns:    vrho to get mdot (rho = f(llm))
                                 vrho- means change rows + cols vrho to get mdot (rho = f(ts, llm))
        :param mu:
        :return:
        '''
        # r_s_for_t_l_vrho = '', 't', 'l', 'lm', 'vrho'


        fname = 't_' + l_or_lm + '_vrho'
        t_llm_vrho = Save_Load_tables.load_table(fname, 't', l_or_lm, '_vrho', self.set_metal, self.bump)
        vrho = t_llm_vrho[1:,1:]

        # -------------------- --------------------- ----------------------------
        c = np.log10(4 * 3.14 * ((Constants.solar_r) ** 2) / Constants.smperyear)
        mdot = np.zeros((vrho.shape))

        if r_s_for_t_l_vrho == '': #------------------------REQUIRED r_s = float
            c2 = c + np.log10(r_s ** 2)
            mdot = vrho + c2

        if r_s_for_t_l_vrho == 't' or r_s_for_t_l_vrho == 'ts': # ---r_s = 1darray
            for i in range(vrho[:,0]):
                mdot[i,:] = vrho[i,:] + c + np.log10(r_s[i] ** 2)

        if r_s_for_t_l_vrho == 'l' or r_s_for_t_l_vrho == 'lm': # ---r_s = 1darray
            for i in range(vrho[0,:]):
                mdot[:,i] = vrho[:,i] + c + np.log10(r_s[i] ** 2)

        if r_s_for_t_l_vrho == 'vrho': #---------------------REQUIRED r_s = 2darray
            cols = len(vrho[0, :])
            rows = len(vrho[:, 0])
            m_dot = np.zeros((rows, cols))

            for i in range(rows):
                for j in range(cols):
                    m_dot[i, j] = vrho[i, j] + c + np.log10(r_s[i, j] ** 2)

        global t_llm_mdot
        t_llm_mdot = Math.combine(t_llm_vrho[0,1:], t_llm_vrho[1:,0], mdot)
        Save_Load_tables.save_table(t_llm_mdot, self.set_metal, 't_' + l_or_lm + '_mdot', 't', l_or_lm, 'mdot', self.out_dir)

    def plot_t_rho_kappa(self, metal, bump, ax=None, fsz=12):

        show_plot = False
        if ax == None:  # if the plotting class is not given:
            fig = plt.figure()
            # fig.subplots_adjust(hspace=0.2, wspace=0.3)
            ax = fig.add_subplot(1, 1, 1)
            show_plot = True

        t_rho_k = Save_Load_tables.load_table('t_rho_k','t','rho','k', metal, bump, self.out_dir)

        PlotBackground.plot_color_background(ax, t_rho_k, 't', 'rho', 'k', metal, 'z:{}'.format(metalicity(metal)), 1.0, True, 12, 0)


        if show_plot:
            ax.text(0.95, 0.05, 'PRELIMINARY', fontsize=50, color='gray', ha='right', va='bottom', alpha=0.5)
            ax.legend(bbox_to_anchor=(1, 0), loc='lower right', ncol=1, fontsize=fsz)
            plot_name = './t_rho_k__{}_{}.pdf'.format(self.bump, metal)
            ax.set_xlabel(Labels.lbls('t'), fontsize=fsz)
            ax.set_ylabel(Labels.lbls('rho'), fontsize=fsz)
            # plt.grid()
            plt.xticks(fontsize=fsz)
            plt.yticks(fontsize=fsz)
            plt.savefig(plot_name)
            plt.show()
        else:
            return ax


if __name__ == "__main__":


    t_fe_bump1 = 5.19  # 5.21 # 5.18
    t_fe_bump2 = 5.45
    t_he2_bump1 = 4.65
    t_he2_bump2 = 5.00

    #
    fname = "./data/OPAL/table8.data"
    #
    # Read_Table(fname)
    make = OPAL_work('gal', 'Fe', 1000, False) # False to load the cases-limits
    make.set_plots_clean = True

    make.save_t_k_rho(3.2, None, 1000)
    make.save_t_rho_k(None, None, 4.1, 5.6)
    make.plot_t_rho_kappa("gal", "Fe")
    #



#
