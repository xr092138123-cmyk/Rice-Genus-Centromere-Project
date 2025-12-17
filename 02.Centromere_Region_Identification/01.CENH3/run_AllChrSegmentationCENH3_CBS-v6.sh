/share/org/YZWL/yzwl_liubx/wangdan/pycharm_script/Oryza_Centromere_project/run_AllChrSegmentationCENH3_CBS-v6.R

$Rscript ~/wangdan/pycharm_script/Oryza_Centromere_project/run_AllChrSegmentationCENH3_CBS-v6.R
Usage: 
Options:
        --sample_file=FILE 第一轮cbs后，匹配到原始坐标的segment文件，必选
        --raw_log2_file=FILE 原始的log2Ratio的信号值，必选
        --known_cen_file=FILE 人工判读的着丝粒坐标文件，必选
        --prefix=STRING  输出目录，必选
        --known_cen_filename_filter  读取人工判读着丝粒文件的哪个单倍型，需与人工判读文件一致，必选
        --haplotype_suffix  用于画原始的散点图，选择单倍型，默认：hap1
        --plot_segmented_points 是否画segment图（中间文件） 默认：TRUE
        --undo_sd_vec=VALUES 以逗号分隔的undo.SD值列表，默认：1
        --cen_logratio_threshold  着丝粒推断的log2ratio阈值，默认：0.5
        --max_gap_points=INTEGER 着丝粒推断中低信号区域允许的最大间隙点数，默认：20
        --min_peak_points=INTEGER 着丝粒推断中，segment被视为有效峰区域所需的最小数量，默认：1
        --chromosomes=CHR_LIST 以逗号分隔的目标染色体列表。使用'ALL'或输入文件中找到的所有染色体，默认："ALL"，或者传入"Chr01,Chr02"
        --multi_cen_min_raw_block_points=INTEGER 富集度——在原始块中(所有点>cen_logratio_threshold)被视为多着丝粒检测候选的最小连续数据点数，设置为0表示只找一个着丝粒区域；色设置成其他值，则表示考虑周围多少个散点的信息，如30；默认：50
        --help
		
		
运行命令：
Rscript ~/wangdan/pycharm_script/Oryza_Centromere_project/run_AllChrSegmentationCENH3_CBS-v6.R --sample_file GG_Omey_hap1.sample1.CENH3.segmentation.bdg --raw_log2_file ~/Oryza_Centromere/02.ChIP-seq_CENH3/02.CENH3_bdg_analysis/GG_Omey/GG_Omey_Sample1/GG_Omey_1_cenH3_log2ratio_2k.bdg --known_cen_file ~/Oryza_Centromere/tmmp/00.genome_for_centromere/00.Centromere\ regions.xlsx --prefix GG_Omey_hap1 --known_cen_filename_filter GG_Omey_hap1