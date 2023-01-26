import numpy as np
from matplotlib import pyplot as plt
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--n_round', type=int, default=1000)
parser.add_argument('--n_iter', type=int, default=1)

args = parser.parse_args()
n_round = args.n_round
x = np.arange(0, n_round)

outfile_opvi_static = np.load('./bnn_res_mnist/OPVI_gd_static_20.npz')
outfile_opvi_sub = np.load('./bnn_res_mnist/OPVI_gd_sub_20.npz')
outfile_svgd_mini = np.load('./bnn_res_mnist/SVGD_gd_static_20.npz')
outfile_svgd_full = np.load('./bnn_res_mnist/SVGD_gd_static_10000.npz')
outfile_ld_mini = np.load('./bnn_res_mnist/LD_gd_static_20.npz')
outfile_ld_full = np.load('./bnn_res_mnist/LD_gd_static_10000.npz')
outfile_ld_sub = np.load('./bnn_res_mnist/LD_gd_sub_100.npz')
outfile_opvi_sub_nop = np.load('./bnn_res_mnist/OPVI_gd_sub_20_nop.npz')


T_acc_opvi_static = outfile_opvi_static['T_acc'].flatten()
L_time_opvi_static = outfile_opvi_sub['L_time'].flatten()
T_acc_opvi_sub = outfile_opvi_sub['T_acc'].flatten()
L_time_opvi_sub = outfile_opvi_sub['L_time'].flatten()
T_acc_svgd_mini = outfile_svgd_mini['T_acc'].flatten()
L_time_svgd_mini = outfile_svgd_mini['L_time'].flatten()
T_acc_svgd_full = outfile_svgd_full['T_acc'].flatten()
L_time_svgd_full = outfile_svgd_full['L_time'].flatten()
T_acc_ld_mini = outfile_ld_mini['T_acc'].flatten()
L_time_ld_mini = outfile_ld_mini['L_time'].flatten()
T_acc_ld_full = outfile_ld_full['T_acc'].flatten()
L_time_ld_full = outfile_ld_full['L_time'].flatten()

T_acc_ld_sub = outfile_ld_sub['T_acc'].flatten()
T_acc_opvi_sub_nop = outfile_opvi_sub_nop['T_acc'].flatten()
print(f'time opvi: {L_time_opvi_sub} time full batch ld : {L_time_ld_full}')

fig, ax = plt.subplots(dpi=400, figsize = (5, 5))

# print(f'X shape {x.shape} Y_acc shape {T_acc_opvi_static.shape} Y_time shape {L_time_opvi_static.shape}')
# learning curve
# ax.plot(x, T_acc_opvi_static, label=r'OPVI $B =20$', linewidth=1.5)
ax.plot(x, T_acc_ld_mini, label=r'LD $B =20$', linewidth=1)
ax.plot(x, T_acc_ld_full, label=r'LD $B =10k$', linewidth=1)
ax.plot(x, T_acc_svgd_mini, label=r'SVGD $B =20$', linewidth=1)
ax.plot(x, T_acc_svgd_full, label=r'SVGD $B =10k$', linewidth=1)
ax.plot(x, T_acc_opvi_static, label=r'OPVI $B =20$', linewidth=1)
ax.plot(x, T_acc_opvi_sub, label=r'OPVI $B_t =t^{0.55}$', color='red', linewidth=1)
# ax.plot(x, T_acc_ld_sub, label=r'LD $B_t =t^{0.55}$', linewidth=1)
# ax.plot(x, T_acc_opvi_sub_nop, label=r'OPVI(no-prior) $B_t =t^{0.55}$', linewidth=1)
ax.set(xlabel='t', ylabel='Accuracy')
# ax.set_title('learning curve', fontsize=25)
ax.set_xlim([0, 500])
ax.grid()
plt.legend()
plt.show()
plt.savefig('test_1.png')

# # time curve
# ax[1].plot(x, L_time_opvi_static)
# ax[1].plot(x, L_time_opvi_sub)
# ax[1].set_title('time')