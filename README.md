# Rice-Genus-Centromere-Project


This repository contains the complete scripts and configuration files for a comprehensive **Rice-Genus-Centromere-Project**, covering assembly quality assessment, structural feature analysis (repeats, TEs, rDNA, organelle DNA transfer), phylogeny, epigenetics (Methylation), and HiC data processing.

01. Quality Assessment
 [01.busco.sh](./01.quality_assessment/01.busco.sh) — Run BUSCO to assess the completeness of the genome assembly.
 [02.omark.py](./01.quality_assessment/02.omark.py) — Assess the evolutionary plausibility and taxonomic consistency of the predicted protein-coding gene set.
 [0301.mapping_reads2genome.py](./01.quality_assessment/03.sequencing_depth/0301.mapping_reads2genome.py) — Map raw HiFi/ONT sequencing reads back to the assembled genome.
 [0302.cal_sequencing_depth.py](./01.quality_assessment/03.sequencing_depth/0302.cal_sequencing_depth.py) — Calculate whole-genome sequencing depth and coverage uniformity based on mapping results.
 [04.QV.py](./01.quality_assessment/04.QV.py) — Calculate Genome Quality Value (QV) based on kmer spectrum.

02. Segment Value (Sequence Segment Value Analysis)
  [analyze_moddotplot.py](./02.segment_value/analyze_moddotplot.py) — Parse moddotplot results, calculate sequence self-similarity and repeat characteristics across different window scales.
  [aggregate_repeats_results.py](./02.segment_value/aggregate_repeats_results.py) — Aggregate repeat statistics from different windows and centromere regions.
  [analyze_centromere_dynamics.py](./02.segment_value/analyze_centromere_dynamics.py) — Dynamically select the optimal analysis window for each centromere and classify structural types.
  [extract_top1_seqs.py](./02.segment_value/extract_top1_seqs.py) — Extract the most dominant (Top1) repeat sequence under the optimal window.
  [calculate_at_content_top1_vs_genome.py]./02.segment_value/calculate_at_content_top1_vs_genome.py) — Compare the AT content difference between the centromere Top1 repeat and the whole genome.
  [plot_top1_dominance.py](./02.segment_value/plot_top1_dominance.py) — Plot a heatmap of Top1 repeat dominance across different centromeres.
  [generate_csub_scripts.py](./02.segment_value/generate_csub_scripts.py) — Generate batch job submission scripts for cluster environments.
  [1.Segment_Value_Based_Identification_of_Centromeric_Intervals.txt](./02.segment_value/1.Segment_Value_Based_Identification_of_Centromeric_Intervals.txt) — This module processes centromeric regions identified in the genome assembly using the moddotplot tool. The goal is to analyze sequence self-similarity and identify the most dominant higher-order repeat (Top1 sequence) by dynamically assessing different sliding window sizes.

03. SV Between Haplotypes (Structural Variation) [01.SV_between_haplotypes.sh](./03.SV_between_haplotypes/01.SV_between_haplotypes.sh) — Complete main pipeline script for detecting structural variations between haplotypes.

04. Satellite Annotation
  [01.run_TRASH.sh](./04.satellite_annotation/01.run_TRASH.sh) — Run the TRASH tool to identify tandem repeats across the whole genome.
  [02.constrain_the_length.py](./04.satellite_annotation/02.constrain_the_length.py) — Classify satellite DNAs based on repeat unit length.
  [03.filter_numbers.py](./04.satellite_annotation/03.filter_numbers.py) — Filter satellite DNA results with too few copy numbers.
  [04.cal_edit_distance.py]./04.satellite_annotation/04.cal_edit_distance.py) — Calculate the minimum cyclic edit distance between satellite DNA repeat units.

05. TE Annotation (Transposable Element) [01.run_HiTE.sh](./05.TE_annotation/01.run_HiTE.sh) — Run HiTE to annotate LTR retrotransposons across the whole genome.
 [02.gff2bed.py](./05.TE_annotation/02.gff2bed.py) — Convert HiTE output GFF3 annotation results to BED format.
 [03.TEsorter_filter.py](./05.TE_annotation/03.TEsorter_filter.py) — Use TEsorter to classify and uniformly name intact LTR transposons.
 [0400.main.py](./05.TE_annotation/04.insertion_time/0400.main.py) — Master control script for LTR insertion time analysis.
 [0401.split_paired_LTR.py](./05.TE_annotation/04.insertion_time/0401.split_paired_LTR.py) — Extract paired left and right LTR sequences from the BED file.
 [0402.insertion_time.py](./05.TE_annotation/04.insertion_time/0402.insertion_time.py) — Calculate sequence divergence and estimate insertion time.
 [0403.add_info.py](./05.TE_annotation/04.insertion_time/0403.add_info.py) — Add sample or genome prefix information to the insertion time results.
 [0404.map_time2bed.py](./05.TE_annotation/04.insertion_time/0404.map_time2bed.py) — Map insertion time information back to the original BED annotation file.

06. Phylogenetic Analysis [01.phylogenetic_analysis.sh](./06.Phylogenetic_analysis/01.phylogenetic_analysis.sh) — Complete pipeline script for phylogenetic analysis.

07. rDNA Annotation (Ribosomal DNA) [01.run_barrnap.py](./07.rDNA_annotation/01.run_barrnap.py) — Run barrnap for automated annotation of rDNA loci in the genome.
 [02.gff2bed.py](./07.rDNA_annotation/02.gff2bed.py) — Convert rDNA annotation results from GFF format to BED format.
 [03.cal_num.py](./07.rDNA_annotation/03.cal_num.py) — Count the copy numbers of different rDNA types on each chromosome.
 [04.heatmap.py](./07.rDNA_annotation/04.heatmap.py) — Plot a heatmap of rDNA copy number distribution across chromosomes.

08. NUMT and NUPT Annotation (Organelle DNA Transfer) [01.run_oatk.sh](./08.NUMT_and_NUPT_annotation/01.run_oatk.sh) — Use oatk to assemble mitochondrial and chloroplast genomes from long-read sequencing data.
 [02.blastn_organelle_to_genome.py](./08.NUMT_and_NUPT_annotation/02.blastn_organelle_to_genome.py) — Automatically run makeblastdb and blastn to align organelle genomes to the nuclear genome.
 [03.filter_blast_results.py](./08.NUMT_and_NUPT_annotation/03.filter_blast_results.py) — Filter high-quality alignment results based on identity and statistical thresholds.
 [04.blast_results2bed.py](./08.NUMT_and_NUPT_annotation/04.blast_results2bed.py) — Convert BLAST alignment results to BED format.
 [05.merge_blastbed.py](./08.NUMT_and_NUPT_annotation/05.merge_blastbed.py) — Merge overlapping BED intervals to avoid double counting.
 [06.cal_length.py](./08.NUMT_and_NUPT_annotation/06.cal_length.py) — Calculate the total length of NUMT and NUPT fragments on each chromosome.

09. HOR (Higher-Order Repeats)HOR Analysis Module Based on Satellite Repeats [01.HOR_detection_and_scoring.py](./09.HOR/windows/01.HOR_detection_and_scoring.py) — Main program for satellite repeat HOR detection and scoring.
 [02.HOR_score_plot.R](./09.HOR/windows/02.HOR_score_plot.R) — Script for visualizing HOR scoring results.
 [1.Batch_extract_Pairwise_Identity.py](./09.HOR/satellite/1.Batch_extract_Pairwise_Identity.py) — Batch extraction of pairwise sequence alignment identity.
 [2.Calculate_HORpairscore_from_Pairwise_Identity.py](./09.HOR/satellite/2.Calculate_HORpairscore_from_Pairwise_Identity.py) — Calculate HOR pair scores from pairwise identity.
 [3.Calculate_HORpairscore_from_Pairwise_Identity.sh](./09.HOR/satellite/3.Calculate_HORpairscore_from_Pairwise_Identity.sh) — Shell script wrapper for running batch HOR score calculation.
 [4.plot_scores_by_blocks.py](./09.HOR/satellite/4.plot_scores_by_blocks.py) — Visualize HOR scores by blocks.

10. Methylation Analysis (DNA Methylation) [01.quality_control.sh](./10.methylation_analysis/01.quality_control.sh) — Perform quality control on raw WGBS data.
 [02.filter.sh](./10.methylation_analysis/02.filter.sh) — Further filter sequencing data to remove low-quality reads and potential contaminant sequences.
 [03.bismark_index.sh](./10.methylation_analysis/03.bismark_index.sh) — Build the reference genome bisulfite alignment index using Bismark.
 [04.bismark_bowtie2.sh](./10.methylation_analysis/04.bismark_bowtie2.sh) — Run Bismark based on Bowtie2 to align WGBS data to the reference genome.
 [05.bismark_deduplicate.sh](./10.methylation_analysis/05.bismark_deduplicate.sh) — Perform deduplication on Bismark alignment results.
 [06.WGBS_mapping_stat.py](./10.methylation_analysis/06.WGBS_mapping_stat.py) — Statistically summarize WGBS mapping results and coverage characteristics.
 [07.bismark_methylation_extractor.sh](./10.methylation_analysis/07.bismark_methylation_extractor.sh) — Extract methylation information for CpG, CHG, and CHH sites.
 [08.bismark_CX2methykit.pl](./10.methylation_analysis/08.bismark_CX2methykit.pl) — Convert Bismark output to the input format recognizable by methylKit.
 [09.methylation_level_calculation.R](./10.methylation_analysis/09.methylation_level_calculation.R) — Calculate DNA methylation levels in different genomic regions or contexts using methylKit.
 Visualization Scripts [01.boxplot.R](./10.methylation_analysis/10.plot/01.boxplot.R) — Plot box plots of DNA methylation levels.
 [02.lineplot.R](./10.methylation_analysis/10.plot/02.lineplot.R) — Plot line plots of DNA methylation levels.

12. HiC Data Processing [01.run_HiCpro.sh](./11.HiC_data_processing/01.run_HiCpro.sh) — Complete main pipeline script for processing HiC data using HiCPro.
 [02.process_HiC_matrix.py](./11.HiC_data_processing/02.process_HiC_matrix.py) — Process the contact matrix output by HiCPro.
 [confighicpro.txt](./11.HiC_data_processing/confighicpro.txt) — HiCPro run configuration file.



