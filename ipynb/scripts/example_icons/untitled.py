fit_res, icc_ = coh_plots._fit_models_no_filt(example_data, models, tech_order)
sub_data = sim_values.loc[example_data['cohort'] == 'cohort_14'].copy()
sub_data.drop(columns=['cohort_slope'], inplace=True)
cohort_data = example_data.loc[example_data['cohort'] == 'cohort_14']
figa = plt.figure(dpi=150, 
                  constrained_layout=False,
                  figsize=(6, np.sum([0.75, 2.5, 2, 1.25])),
                  facecolor='w')
fig_seed, fig_ind, fig_dep, fig_c = \
    figa.subfigures(4, 1, height_ratios=[0.75, 2.5, 2, 1.25])
fig_dem, fig_exp1 = fig_ind.subfigures(1, 2, width_ratios=[3.5, 2.5], facecolor='None')
figdl, fig_dm = fig_dem.subfigures(2, 1, height_ratios=[1.5, 1], facecolor='None')
fig_exp, fig_rand = fig_exp1.subfigures(2, 1, height_ratios=[1.5, 1])
fig_cl, fig_cr = fig_c.subfigures(1, 2, width_ratios=[4, 2])
figs, figw, _ = figa.subfigures(3, 1, height_ratios=[0.75, 4.5, 1.25])

coh_plots.summarize_simulation_parameters(fig_seed, [1, 1, 1, 1], -0.5, 0, example_icc, eff_mod=None)

exp_plots.build_sim_process_ref(figdl, cohort_data)
exp_plots._build_table_plot(fig_dm, cohort_data, n_head=3)

exp_plots._select_exposure(fig_exp, cohort_data, smoking_effect)
exp_plots.illustrate_exposure_res(fig_rand, cohort_data)

exp_plots.build_sim_sum_icons(fig_dep, sub_data)

fig_cl.set_facecolor('None')
exp_plots.plot_example_cohort(fig_cl, example_data, 'cohort_06')
fig_cr.set_facecolor('None')
exp_plots.pooled_res_plot(fig_cr, fit_res=fit_res, icc=icc_)

axw = figw.add_subplot(1,1,1, facecolor='None')
axw.xaxis.set_tick_params(**exp_plots.hide_axes)
axw.yaxis.set_tick_params(**exp_plots.hide_axes)
axw.set_position((0, 0.025, 1, 0.975))
axw.set_xlim(0, 1)
axw.set_ylim(0, 1)
axw.set_yticks([0, 0.9825])
axw.set_xticks([0, 1])
axw.text(0.03, 1, 'For the $i$th ICS:', size=12, va='top', ha='left')
axw.plot([0, 0.02], [0.9825, 0.9825], 'k-', linewidth=1)
axw.plot([1, 0.25], [0.9825, 0.9825], 'k-', linewidth=1)
sn.despine(ax=axw, left=False, right=False, bottom=False, top=True, trim=True)

fig_seed.text(0.1, 1, 'A', size=12, ha='left', va='top')
fig_dem.text(0.01, 0.95, 'B', size=12, ha='left', va='top')
fig_exp1.text(0, 0.95, 'C', size=12, ha='left', va='top')
fig_dep.text(0.01, 1, 'D', size=12, ha='left', va='top')
fig_cl.text(0.01, 1, 'E', size=12, ha='left', va='top')
fig_cr.text(0.0825, 1, 'F', size=12, ha='left', va='top')

#Saves figure
os.makedirs('output/png', exist_ok=True)
os.makedirs('output/pdf', exist_ok=True)
figa.savefig('output/png/simulation_schematic.png', dpi=300, facecolor='w')
figa.savefig('output/pdf/simulation_schematic.pdf', dpi=300, facecolor='None')

figa