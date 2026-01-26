import pathlib
import pandas as pd
import numpy as np
from collections import defaultdict
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Spacer, Paragraph, PageBreak, KeepTogether

import analysis_setup, naming, utils

data_dir = pathlib.Path('data')

# get eye tracker info
et_info = utils.get_et_info_from_recordings(data_dir, analysis_setup.eye_trackers)

# other setup
custom_cmap     = utils.custom_colormap(analysis_setup.colors)
custom_cmap_fs  = utils.custom_colormap(analysis_setup.colors_fs)


doc = SimpleDocTemplate("report_summary.pdf", pagesize=A4)
elements = []

elements.append(Paragraph("Benchmark report card", utils.CenteredHeading1))
elements.append(Spacer(1, 12))

elements.append(Paragraph("This report card contains an evaluation of the following eye trackers.", utils.styles["BodyText"]))
elements.append(Spacer(1, 3))

# make table with et specs
elements.append(KeepTogether([
    utils.make_apa_table(
        et_info,
        colNames=analysis_setup.et_colnames,
        colWidths=[120, 60, 60, 100, 100],
        decimals=defaultdict(lambda: 2, weight=1, sample_rate=0),
        zebra=True),
    Paragraph("Table 1. Tested eye trackers", utils.APA_TableCaption)]))


# Methodology
elements.append(Paragraph("Methodology", utils.LeftHeading))
elements.append(Paragraph("The eye trackers were evaluated in static (head/glasses fixed, \
                    Station 1) and dynamic (head/glasses moving, Station 2) conditions. See \
                    Niehorster et al. (in prep) for further details.",
                    utils.styles["BodyText"]))

# Methods Figures
# Figure 1
elements.append(KeepTogether([
    Paragraph("Station 1", utils.styles["Heading3"]),
    Paragraph("In Station 1, participants were asked to perform two tasks with \
        their heads placed on a chinrest. In the first task (Figure 1A, B), \
        targets at different locations were fixated one at the time. Accuracy, precision \
        and data loss were computed per target from gaze data for 1-s intervals.",
        utils.styles["BodyText"])]))

elements.append(Paragraph("In the second task, participants were asked to fixated the central target dot \
                 while the stimulus periodically changed from black to white (Figure 1C) to cause large pupil size changes. \
                 Apparent gaze shifts due to pupil size changes were quantified.",
                 utils.styles["BodyText"]))
elements.append(Spacer(1, 4))

elements.append(KeepTogether([utils.auto_image("station1_methods.PNG"),
                              Paragraph("Figure 1. (A) Arrangement of target locations for accuracy, precision \
                                and data loss assessment. (B) The fixation targets were arranged into four different \
                                rings. (C) Stimuli to assess PSA. The screen alternated between \
                                black and white to induce pupil-size changes. The small red dots were used to map \
                                gaze from the eye-tracker's scene camera to the reference frame of the stimulus.",
                                utils.APA_FigureCaption)]))

# Figure 2
elements.append(KeepTogether([
    Paragraph("Station 2", utils.styles["Heading3"]),
    Paragraph("In Station 2, the influence of slippage on the gaze signal provided by the eye tracker \
        was investigated by asking participants to move the glasses about 1 cm in three directions (Figure 2A), \
        Moreover, parallax errors were investigated by viewing target dots at different distances (Figure 2B, C). \
        We operationalized the parallax error as the difference in estimated gaze position between two viewing distances \
        (2 m and 30 cm).",
        utils.styles["BodyText"])]))

elements.append(KeepTogether([utils.auto_image("station2_methods.PNG"),
                              Paragraph("Figure 2. (A) Induced slippage movements. The parallax error is investigated \
                                using fixation grids viewed from (B) 30 cm and (C) 2 m.",
                                utils.APA_FigureCaption
                                )]))


elements.append(KeepTogether([
    Paragraph("Reference", utils.styles["Heading3"]),
    Paragraph("Niehorster, D. C., Marano, G., Pettenella, A., Carminati, M., Melloni, F., Merigo, L. & Nyström, M. (in prep). Methodology for systematically assessing wearable eye tracker performance.",
        utils.styles["BodyText"])]))


# RESULTS
elements.append(PageBreak())
elements.append(Paragraph("Results", utils.LeftHeading))

# Results station 1
if not (data_dir/naming.station1_1).is_file() or not (data_dir/naming.station1_2).is_file():
    print("No data found for Station 1 analysis, skipping...")
else:
    # 1.1 Fixation task
    df_fixation = pd.read_csv(data_dir/naming.station1_1, delimiter='\t')
    D = df_fixation.groupby(by = ['tracker', 'ring']).mean(numeric_only=True)
    D = D[['acc', 'rms', 'std', 'Fs', 'relative_Fs', 'data_loss']].reset_index()
    # get number of participants per tracker
    n_participants = df_fixation.groupby(by=['tracker','pid']).mean(numeric_only=True).groupby(level='tracker').size().to_dict()

    colors = {c: custom_cmap for c in ['acc', 'rms', 'std', 'data_loss']} | {'relative_Fs': custom_cmap_fs}
    colors_limits = utils.get_color_limits(D, {'acc': analysis_setup.fixation_acc_lim, 'rms': analysis_setup.rms_lim, 'std': analysis_setup.std_lim, 'data_loss': analysis_setup.loss_lim, 'relative_Fs': analysis_setup.fs_lim}, ['acc'], analysis_setup.fixation_acc_percentiles)
    first_col = {et.value: (et_info.loc[et,'name'], f'(N={n_participants[et.value]})') for et in et_info.index}

    elements.append(KeepTogether([
        Paragraph("Station 1", utils.styles["Heading3"]),
        Paragraph("Table 2: Accuracy, precision (RMS, STD), and data loss for different rings.", utils.APA_TableCaption),
        utils.make_apa_table(
            D,
            colNames=analysis_setup.colnames,
            colWidths=[120, 30, 50, 50, 50, 50, 50, 50],
            decimals=defaultdict(lambda: 2, ring=0),
            colors=colors,
            colors_limits=colors_limits,
            first_col_levels=first_col,
            body_cell_padding=(6,1.5))]))
    elements.append(Spacer(1, 12))

    # 1.2 PSA
    df_PSA = pd.read_csv(data_dir/naming.station1_2, delimiter='\t')
    D = df_PSA.groupby(by = ['tracker', 'target location']).mean(numeric_only=True)
    D = D[['offset_total', 'pd_diff', 'Fs', 'relative_Fs', 'data_loss']].reset_index()
    # get number of participants per tracker
    n_participants = df_PSA.groupby(by=['tracker','pid']).mean(numeric_only=True).groupby(level='tracker').size().to_dict()

    colors = {c: custom_cmap for c in ['offset_total', 'data_loss']} | {'relative_Fs': custom_cmap_fs}
    colors_limits = utils.get_color_limits(D, {'offset_total': analysis_setup.PSA_acc_lim, 'data_loss': analysis_setup.loss_lim, 'relative_Fs': analysis_setup.fs_lim}, ['offset_total'], analysis_setup.PSA_acc_percentiles)
    first_col = {et.value: (et_info.loc[et,'name'], f'(N={n_participants[et.value]})') for et in et_info.index}

    elements.append(KeepTogether([
        Paragraph("Table 3: Apparent gaze shift due to pupil size changes (averaged across participants).", utils.APA_TableCaption),
        utils.make_apa_table(
            D,
            colNames=analysis_setup.colnames,
            colWidths=[120, 50, 50, 50, 50, 50, 50],
            colors=colors,
            colors_limits=colors_limits,
            first_col_levels=first_col,
            body_cell_padding=(6,1.5))]))


# Results station 2
if not (data_dir/naming.station2_1).is_file() or not (data_dir/naming.station2_2).is_file():
    print("No data found for Station 2 analysis, skipping...")
else:
    # 2.1 Slippage
    df_slippage = pd.read_csv(data_dir/naming.station2_1, delimiter='\t')
    D = df_slippage.groupby(by = ['tracker', 'trial']).mean(numeric_only=True)
    D = D[['shift_x', 'shift_y', 'Fs', 'relative_Fs', 'data_loss']].reset_index()
    D = D.replace(analysis_setup.slippage_trials)
    # get number of participants per tracker
    n_participants = df_slippage.groupby(by=['tracker','pid']).mean(numeric_only=True).groupby(level='tracker').size().to_dict()

    colors = {c: custom_cmap for c in ['shift_x', 'shift_y', 'data_loss']} | {'relative_Fs': custom_cmap_fs}
    colors_limits = utils.get_color_limits(D, {'shift_x': analysis_setup.slippage_acc_lim, 'shift_y': analysis_setup.slippage_acc_lim, 'data_loss': analysis_setup.loss_lim, 'relative_Fs': analysis_setup.fs_lim}, ['shift_x','shift_y'], analysis_setup.slippage_acc_percentiles)
    first_col = {et.value: (et_info.loc[et,'name'], f'(N={n_participants[et.value]})') for et in et_info.index}

    elements.append(KeepTogether([
        Paragraph("Station 2", utils.styles["Heading3"]),
        Paragraph("Table 4: Apparent gaze shift due to glasses slippage in different directions.", utils.APA_TableCaption),
        utils.make_apa_table(
            D,
            colNames=analysis_setup.colnames,
            colWidths=[120, 80, 50, 50, 50, 50, 50, 50],
            colors=colors,
            colors_limits=colors_limits,
            first_col_levels=first_col,
            body_cell_padding=(6,1.5))]))
    elements.append(Spacer(1, 12))

    # 2.2 parallax
    diff_dists = [analysis_setup.parallax_distances[0], analysis_setup.parallax_distances[-1]]
    cols = ['shift_x', 'shift_y']
    df_parallax = pd.read_csv(data_dir/naming.station2_2, delimiter='\t')
    # this needs to be done carefully to make sure we don't average the wrong signed values
    # first get mean per participant, tracker, distance, target (so averaging after multiple looks at targets)
    D = df_parallax.groupby(by = ['pid', 'tracker', 'distance', 'target_id']).mean(numeric_only=True).reset_index(level='distance')
    # compute parallax between the two distances
    D2 = D[D['distance']==diff_dists[0]][cols] - D[D['distance']==diff_dists[1]][cols]
    # for other measures, average over distances
    D  = D[['Fs', 'relative_Fs', 'data_loss']].reset_index().groupby(by = ['pid', 'tracker', 'target_id']).mean(numeric_only=True)
    # put back in the parallax values
    D[cols] = D2[cols]
    # compute overall shift
    D['shift'] = np.hypot(D['shift_x'], D['shift_y'])
    # now average over targets and participants
    D = D.groupby(by = ['tracker']).mean(numeric_only=True)
    D = D[['shift', 'shift_x', 'shift_y', 'Fs', 'relative_Fs', 'data_loss']].reset_index()
    # get number of participants per tracker
    n_participants = df_parallax.groupby(by=['tracker','pid']).mean(numeric_only=True).groupby(level='tracker').size().to_dict()

    colors = {c: custom_cmap for c in ['shift', 'shift_x', 'shift_y', 'data_loss']} | {'relative_Fs': custom_cmap_fs}
    colors_limits = utils.get_color_limits(D, {'shift': analysis_setup.parallax_acc_lim, 'shift_x': analysis_setup.parallax_acc_lim, 'shift_y': analysis_setup.parallax_acc_lim, 'data_loss': analysis_setup.loss_lim, 'relative_Fs': analysis_setup.fs_lim}, ['shift', 'shift_x', 'shift_y'], analysis_setup.parallax_acc_percentiles, need_abs=['shift_x', 'shift_y'])
    first_col = {et.value: f'{et_info.loc[et,"name"]}\n(N={n_participants[et.value]})' for et in et_info.index}

    elements.append(KeepTogether([
        Paragraph(f"Table 5. Parallax errors (apparent gaze shifts between viewing \
                    distances {diff_dists[0]} cm and {diff_dists[1]} cm)", utils.APA_TableCaption),
        utils.make_apa_table(
            D,
            colNames=analysis_setup.colnames,
            colWidths=[120, 50, 50, 50, 50, 50, 50],
            colors=colors,
            colors_limits=colors_limits,
            first_col_levels=first_col,
            body_cell_padding=(6,1.5))]))

# build PDF
doc.build(elements)