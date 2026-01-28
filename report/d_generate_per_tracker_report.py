import pathlib
import pandas as pd
import numpy as np
from collections import defaultdict
from reportlab.lib.pagesizes import A4
from reportlab.platypus import Spacer, Paragraph, PageBreak, KeepTogether

import analysis_setup, naming, utils

data_dir = pathlib.Path('data')
plot_dir = pathlib.Path('figures')

# get eye tracker info
et_info = utils.get_et_info_from_recordings(data_dir, analysis_setup.eye_trackers)

# other setup
custom_cmap     = utils.custom_colormap(analysis_setup.colors)
custom_cmap_fs  = utils.custom_colormap(analysis_setup.colors_fs)

for et in et_info.index:
    et_nm = et.value

    doc = utils.BookmarkedDocTemplate(f"report_{et_nm}.pdf", pagesize=A4)
    elements = []

    elements.append(Paragraph(f"Detailed performance report for {et_nm}", utils.CenteredHeading1))
    elements.append(Spacer(1, 12))

    elements.append(Paragraph("This report contains an evaluation of the following eye tracker.", utils.styles["BodyText"]))
    elements.append(Spacer(1, 3))

    # make table with et specs
    elements.append(KeepTogether([
        utils.make_apa_table(
            et_info.loc[et],
            colNames=analysis_setup.et_colnames,
            colWidths=analysis_setup.et_colwidths,
            decimals=defaultdict(lambda: 2, weight=1, sample_rate=0),
            zebra=True),
        Paragraph("Table 1. Tested eye tracker", utils.APA_TableCaption)]))


    elements.append(KeepTogether([
        Paragraph("Reference", utils.styles["Heading3"]),
        Paragraph("Niehorster, D. C., Marano, G., Pettenella, A., Carminati, M., Melloni, F., Merigo, L. & Nyström, M. (in prep). Methodology for systematically assessing wearable eye tracker performance.",
            utils.styles["BodyText"])]))


    # RESULTS
    elements.append(Paragraph("Results", utils.LeftHeading))

    # Results station 1
    if not (data_dir/naming.station1_1).is_file() or not (data_dir/naming.station1_2).is_file():
        print("No data found for Station 1 analysis, skipping...")
    else:
        # 1.1 Fixation task
        elements.append(KeepTogether([
            Paragraph("Station 1, Fixation task", utils.styles["Heading3"]),
            utils.auto_image(plot_dir / f'{naming.station1_1_prefix}{et_nm}.png'),
            Paragraph(
                "Data quality metrics (accuracy, RMS-S2S and STD precision, and data loss) per target for the fixation task.",
                utils.APA_FigureCaption
            )]))

        df_fixation = pd.read_csv(data_dir/naming.station1_1, delimiter='\t')
        D = df_fixation[df_fixation.tracker==et_nm].groupby(by = ['pid','ring']).mean(numeric_only=True)
        D = D[['acc', 'rms', 'std', 'Fs', 'relative_Fs', 'data_loss']].reset_index()

        colors = {c: custom_cmap for c in ['acc', 'rms', 'std', 'data_loss']} | {'relative_Fs': custom_cmap_fs}
        colors_limits = utils.get_color_limits(D, {'acc': analysis_setup.fixation_acc_lim, 'rms': analysis_setup.rms_lim, 'std': analysis_setup.std_lim, 'data_loss': analysis_setup.loss_lim, 'relative_Fs': analysis_setup.fs_lim}, ['acc'], analysis_setup.fixation_acc_percentiles)

        elements.append(KeepTogether([
            Paragraph("Table 2: Accuracy, precision (RMS, STD), and data loss for different rings, per participant.", utils.APA_TableCaption),
            utils.make_apa_table(
                D,
                colNames=analysis_setup.colnames,
                colWidths=analysis_setup.colwidths,
                decimals=defaultdict(lambda: 2, ring=0),
                colors=colors,
                colors_limits=colors_limits,
                first_col_levels={},
                body_cell_padding=(6,1.5))]))
        elements.append(PageBreak())

        # 1.2 PSA
        elements.append(KeepTogether([
            Paragraph("Station 1, PSA", utils.styles["Heading3"]),
            utils.auto_image(plot_dir / f'{naming.station1_2_prefix}{et_nm}.png', scale=.6),
            Paragraph(
                "PSA per target, for each participant.",
                utils.APA_FigureCaption
            )]))

        df_PSA = pd.read_csv(data_dir/naming.station1_2, delimiter='\t')
        D = df_PSA[df_PSA.tracker==et_nm].groupby(by = ['pid', 'target location']).mean(numeric_only=True)
        D = D[['offset_total', 'pd_diff', 'Fs', 'relative_Fs', 'data_loss']].reset_index()

        colors = {c: custom_cmap for c in ['offset_total', 'data_loss']} | {'relative_Fs': custom_cmap_fs}
        colors_limits = utils.get_color_limits(D, {'offset_total': analysis_setup.PSA_acc_lim, 'data_loss': analysis_setup.loss_lim, 'relative_Fs': analysis_setup.fs_lim}, ['offset_total'], analysis_setup.PSA_acc_percentiles)

        elements.append(KeepTogether([
            Paragraph("Table 3: Apparent gaze shift due to pupil size changes, per participant and target position.", utils.APA_TableCaption),
            utils.make_apa_table(
                D,
                colNames=analysis_setup.colnames,
                colWidths=analysis_setup.colwidths,
                colors=colors,
                colors_limits=colors_limits,
                first_col_levels={},
                body_cell_padding=(6,1.5))]))
        elements.append(Spacer(1, 12))

        explanation = Paragraph(
            "The following plots show, for each participant, the horizontal (x) and vertical (y) gaze offset \
            over time while the screen brightness changes between bright (yellow vertical stripes) and dark \
            (gray vertical stripes). Participants were asked to look at a target on the right, left, and middle \
            of the screen. Negative values correspond to leftward (x) or upward (y) offsets, whereas positive \
            values to downward (x) or rightward (y) offsets.", utils.styles["BodyText"])
        explanation.keepWithNext = 1
        elements.append(explanation)
        elements.append(Spacer(1, 3))

        for pid in df_PSA.pid.unique():
            elements.append(utils.auto_image(plot_dir / f'{naming.station1_2_prefix}{et_nm}_{pid}.png'))
        elements.append(PageBreak())


    # Results station 2
    if not (data_dir/naming.station2_1).is_file() or not (data_dir/naming.station2_2).is_file():
        print("No data found for Station 2 analysis, skipping...")
    else:
        # 2.1 Slippage
        df_slippage = pd.read_csv(data_dir/naming.station2_1, delimiter='\t')
        D = df_slippage[df_slippage.tracker==et_nm].groupby(by = ['pid', 'trial']).mean(numeric_only=True)
        D = D[['shift_x', 'shift_y', 'Fs', 'relative_Fs', 'data_loss']].reset_index()
        D = D.replace(analysis_setup.slippage_trials).rename(columns={'trial': 'slippage direction'})

        colors = {c: custom_cmap for c in ['shift_x', 'shift_y', 'data_loss']} | {'relative_Fs': custom_cmap_fs}
        colors_limits = utils.get_color_limits(D, {'shift_x': analysis_setup.slippage_acc_lim, 'shift_y': analysis_setup.slippage_acc_lim, 'data_loss': analysis_setup.loss_lim, 'relative_Fs': analysis_setup.fs_lim}, ['shift_x','shift_y'], analysis_setup.slippage_acc_percentiles)

        elements.append(KeepTogether([
            Paragraph("Station 2, Slippage", utils.styles["Heading3"]),
            Paragraph("Table 4: Apparent gaze shift due to glasses slippage in different directions, per participant.", utils.APA_TableCaption),
            utils.make_apa_table(
                D,
                colNames=analysis_setup.colnames,
                colWidths=analysis_setup.colwidths,
                colors=colors,
                colors_limits=colors_limits,
                first_col_levels={},
                body_cell_padding=(6,1.5))]))
        elements.append(Spacer(1, 12))

        explanation = Paragraph(
            "The following plots show, for each participant, the horizontal (x) and vertical (y) gaze \
            offset over time while the eye-tracker glasses are moved back-and-fourth, side-to-side, \
            or up-and-down.", utils.styles["BodyText"])
        explanation.keepWithNext = 1
        elements.append(explanation)
        elements.append(Spacer(1, 3))

        for pid in df_slippage.pid.unique():
            elements.append(utils.auto_image(plot_dir / f'{naming.station2_1_prefix}{et_nm}_{pid}.png'))
        elements.append(PageBreak())

        # 2.2 parallax
        elements.append(KeepTogether([
            Paragraph("Station 2, Parallax", utils.styles["Heading3"]),
            utils.auto_image(plot_dir / f'{naming.station2_2_prefix}{et_nm}.png', scale=.6),
            Paragraph(
                "Parallax per target, for each participant (orange lines) along with average (red arrow).",
                utils.APA_FigureCaption
            )]))

        diff_dists = [analysis_setup.parallax_distances[0], analysis_setup.parallax_distances[-1]]
        cols = ['shift_x', 'shift_y']
        df_parallax = pd.read_csv(data_dir/naming.station2_2, delimiter='\t')

        # as per summary card, this is done in multiple steps
        D = df_parallax[df_parallax.tracker==et_nm].groupby(by = ['pid', 'distance', 'target_id']).mean(numeric_only=True).reset_index(level='distance')
        # compute parallax between the two distances
        D2 = D[D['distance']==diff_dists[0]][cols] - D[D['distance']==diff_dists[1]][cols]
        # for other measures, average over distances
        D  = D[['Fs', 'relative_Fs', 'data_loss']].reset_index().groupby(by = ['pid', 'target_id']).mean(numeric_only=True)
        # put back in the parallax values
        D[cols] = D2[cols]
        # compute overall shift
        D['shift'] = np.hypot(D['shift_x'], D['shift_y'])
        # now average over targets
        D = D.groupby(by = ['pid']).mean(numeric_only=True)
        D = D[['shift', 'shift_x', 'shift_y', 'Fs', 'relative_Fs', 'data_loss']].reset_index()

        colors = {c: custom_cmap for c in ['shift', 'shift_x', 'shift_y', 'data_loss']} | {'relative_Fs': custom_cmap_fs}
        colors_limits = utils.get_color_limits(D, {'shift': analysis_setup.parallax_acc_lim, 'shift_x': analysis_setup.parallax_acc_lim, 'shift_y': analysis_setup.parallax_acc_lim, 'data_loss': analysis_setup.loss_lim, 'relative_Fs': analysis_setup.fs_lim}, ['shift', 'shift_x', 'shift_y'], analysis_setup.parallax_acc_percentiles, need_abs=['shift_x', 'shift_y'])

        elements.append(KeepTogether([
            Paragraph(f"Table 5. Parallax errors (apparent gaze shifts between viewing \
                        distances {diff_dists[0]} cm and {diff_dists[1]} cm), per participant", utils.APA_TableCaption),
            utils.make_apa_table(
                D,
                colNames=analysis_setup.colnames,
                colWidths=analysis_setup.colwidths,
                colors=colors,
                colors_limits=colors_limits,
                first_col_levels={},
                body_cell_padding=(6,1.5))]))

        explanation = Paragraph(
            f"The following plots show, for each participant, the horizontal (x) and vertical (y) gaze \
            offset for each fixation target at each viewing distance, along with the parallax error \
            (apparent gaze shifts between viewing distances {diff_dists[0]} cm and {diff_dists[1]} cm).", utils.styles["BodyText"])
        explanation.keepWithNext = 1
        elements.append(explanation)
        elements.append(Spacer(1, 3))

        for pid in df_slippage.pid.unique():
            elements.append(utils.auto_image(plot_dir / f'{naming.station2_2_prefix}{et_nm}_{pid}.png'))

    # build PDF
    doc.build(elements)