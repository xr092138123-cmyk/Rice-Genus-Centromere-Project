#!/usr/bin/env Rscript

# Load optparse library for command line argument parsing
suppressPackageStartupMessages(library(optparse))

# Example usage:
# ./run_cen_analysis.R --prefix "GG_Omey_hap2_test" --sample_file "GG/GG_Omey_hap2.sample1.CENH3.segmentation.bdg" --known_cen_filename_filter "GG_Omey_hap2" --haplotype_suffix "hap2" --raw_log2_file "GG/GG_Omey_2_cenH3_log2ratio_2k.bdg"

# Define command line options
option_list <- list(
  make_option(c("-i", "--sample_file"), type="character",default = NULL,
              help="Input segmented BDG file path [such as GG/GG_Omey_hap1.sample1.CENH3.segmentation.bdg]", metavar="FILE"),
  make_option(c("--raw_log2_file"), type="character",default = NULL,
              help="Optional path to BDG file with original (non-segmented) log2ratio values for background plotting [such as GG/GG_Omey_1_cenH3_log2ratio_2k.bdg]", metavar="FILE"),
  make_option(c("-k", "--known_cen_file"), type="character",default = NULL,
              help="Optional path to Excel file with known centromere regions. Leave empty or set to non-existent file to skip. [such as 00.Centromere_regions.xlsx]", metavar="FILE"),
  make_option(c("-p", "--prefix"), type="character",default = NULL,
              help="Output prefix for files and directory [such as GG_Omey_hap1]", metavar="STRING"),
  make_option(c("-f", "--known_cen_filename_filter"), type="character",default = NULL,
              help="Filter for 'filename' column in the known centromere file (used if --known_cen_file is provided and exists) [such as GG_Omey_hap1]", metavar="STRING"),
  make_option(c("--haplotype_suffix"), type="character", default="hap1",
              help="Suffix to append to chromosome names from segmented file (after removing existing _hapN) when looking up in raw_log2_file (e.g., 'hap1', 'hap2'). If empty, uses segmented chromosome name (potentially with its own suffix) directly. [default %default]", metavar="STRING"),
  make_option(c("--plot_segmented_points"), action="store_true", default=TRUE,
              help="If set, plot the scatter points from the segmented input file. Default is not to plot them."),
  make_option(c("-u", "--undo_sd_vec"), type="character", default="1",
              help="Comma-separated list of undo.SD values. Currently only affects avg_plot naming and avg_summary table. [default %default]", metavar="VALUES"),
  make_option(c("-t", "--cen_logratio_threshold"), type="numeric", default=as.numeric(0.5),
              help="Log2ratio threshold for centromere inference [default %default]", metavar="NUMBER"),
  make_option(c("-g", "--max_gap_points"), type="integer", default=as.integer(20),
              help="Maximum allowed gap points in low signal regions for centromere inference [default %default]", metavar="INTEGER"),
  make_option(c("--min_peak_points"), type="integer", default=1,
              help="Minimum number of consecutive data points (after gap filling) required for a peak region to be considered valid for centromere inference. If shorter (and not in multi-CEN mode leading to other valid CENs), the whole chromosome is used. [default %default]", metavar="INTEGER"),
  make_option(c("-c", "--chromosomes"), type="character",
              default="ALL",
              help=sprintf("Comma-separated list of target chromosomes to process. Use 'ALL' or an empty string ('') to process all chromosomes found in the input file. [default: '%s']","ALL" ),
              metavar="CHR_LIST"),
  make_option(c("--multi_cen_min_raw_block_points"), type="integer", default=as.integer(50),
              help="Minimum number of consecutive data points in a raw block (all points > cen_logratio_threshold) to be considered a candidate for multi-centromere detection. If 0, single 'best' centromere mode is used. [default %default]", metavar="INTEGER")
)
#构建参数对象 OptionParser,以便解析
opt_parser=OptionParser(option_list=option_list,
						add_help_option = TRUE,
						prog=NULL,
						epilogue = "Rscript run_AllChrSegmentationCENH3_CBS-v6.R --sample_file GG_Omey_hap1.sample1.CENH3.segmentation.bdg --raw_log2_file GG_Omey_1_cenH3_log2ratio_2k.bdg --known_cen_file 00.Centromere\\ regions.xlsx --prefix GG_Omey_hap1 --known_cen_filename_filter GG_Omey_hap1"
						)

# opt_parser <- OptionParser(option_list=option_list)
opt <- parse_args(opt_parser)

if ( is.null(opt$sample_file) )	{ print_help(opt_parser);q(status=1) }
if ( is.null(opt$raw_log2_file) )	{ print_help(opt_parser);q(status=1) }
if ( is.null(opt$known_cen_file) )	{ print_help(opt_parser);q(status=1) }
if ( is.null(opt$prefix) )	{ print_help(opt_parser);q(status=1) }
if ( is.null(opt$known_cen_filename_filter) )	{ print_help(opt_parser);q(status=1) }

sample_file               <- opt$sample_file
raw_log2_file_path        <- opt$raw_log2_file
haplotype_suffix_for_raw  <- opt$haplotype_suffix
plot_segmented_points     <- opt$plot_segmented_points
prefix                    <- opt$prefix
undo_sd_vec_str           <- opt$undo_sd_vec
undo_sd_vec               <- as.numeric(unlist(strsplit(undo_sd_vec_str, ",")))
if (any(is.na(undo_sd_vec))) {
  stop("Invalid --undo_sd_vec input. Please provide comma-separated numbers (e.g., '1,0.5,2').")
}
cen_logratio_threshold    <- opt$cen_logratio_threshold
max_gap_points            <- opt$max_gap_points
min_peak_points           <- opt$min_peak_points
known_cen_file            <- opt$known_cen_file
known_cen_filename_filter <- opt$known_cen_filename_filter
user_specified_chromosomes_str <- opt$chromosomes
multi_cen_min_raw_block_points <- opt$multi_cen_min_raw_block_points

if(min_peak_points < 1) {
  warning("--min_peak_points must be at least 1. Setting to 1.")
  min_peak_points <- 1
}
if(multi_cen_min_raw_block_points < 0) {
  warning("--multi_cen_min_raw_block_points cannot be negative. Setting to 0 (single centromere mode).")
  multi_cen_min_raw_block_points <- 0
}

cat("── 0. Parameters provided ──────────────────────────────────────────────────\n")
cat(sprintf("Input segmented BDG file: %s\n", sample_file))
cat(sprintf("Plot points from segmented file: %s\n", plot_segmented_points))
cat(sprintf("Raw log2ratio file for plotting: %s\n", if(nzchar(raw_log2_file_path)) raw_log2_file_path else "Not provided"))
if (nzchar(raw_log2_file_path)) {
  cat(sprintf("Haplotype suffix for raw file lookup: '%s' (if empty, uses segmented chr name directly)\n", haplotype_suffix_for_raw))
}
cat(sprintf("Output prefix: %s\n", prefix))
cat(sprintf("Undo SD values: %s (Note: CBS removed, only affects avg_plot loop/naming)\n", paste(undo_sd_vec, collapse=", ")))
cat(sprintf("Centromere log2ratio threshold: %.2f\n", cen_logratio_threshold))
cat(sprintf("Max gap points: %d\n", max_gap_points))
cat(sprintf("Minimum peak points (after gap fill): %d\n", min_peak_points))
cat(sprintf("Multi-centromere mode: %s\n", if(multi_cen_min_raw_block_points > 0) sprintf("Enabled (min raw block points: %d)", multi_cen_min_raw_block_points) else "Disabled (single 'best' centromere mode)"))
cat(sprintf("Known centromere file: %s\n", if(nzchar(known_cen_file)) known_cen_file else "Not provided or empty"))
cat(sprintf("Known centromere filename filter: %s\n", if(nzchar(known_cen_filename_filter)) known_cen_filename_filter else "Not provided or empty"))
cat(sprintf("Chromosomes to process (raw input/default): %s\n", user_specified_chromosomes_str))
cat("───────────────────────────────────────────────────────────────────────────\n\n")

options(repos = "https://mirrors.tuna.tsinghua.edu.cn/CRAN/")
suppressPackageStartupMessages({
  library(data.table)
  library(ggplot2)
  library(readxl)
  library(stringr) # For str_remove
})

output_dir <- file.path(getwd(), prefix)
if (!dir.exists(output_dir)) {
  dir.create(output_dir, recursive = TRUE)
  cat(sprintf("✅ 已创建输出目录: %s\n", output_dir))
} else {
  cat(sprintf("ℹ️  输出目录已存在: %s\n", output_dir))
}

known_cen_df_for_plot <- NULL
if (nzchar(known_cen_file) && file.exists(known_cen_file)) {
  cat(sprintf("ℹ️  读取真实着丝粒文件: %s\n", known_cen_file))
  raw_known <- tryCatch(read_excel(known_cen_file), error = identity)
  if (inherits(raw_known, "error")) {
    warning(sprintf("⚠️  读取真实着丝粒文件失败: %s. 跳过。\n%s", known_cen_file, raw_known$message))
  } else {
    filename_col <- intersect(c("filename", "Filename"),colnames(raw_known))[1]
    chr_col      <- intersect(c("Chr","Chromosome"),colnames(raw_known))[1]
    start_col    <- intersect(c("CENH3-based Start(bp)","CENH3.based.Start.bp","CENH3_based_Start_bp","Start_bp","Start","start"),colnames(raw_known))[1]
    end_col      <- intersect(c("CENH3-based End(bp)","CENH3.based.End.bp","CENH3_based_End_bp","End_bp","End","end"),colnames(raw_known))[1]

    if (any(is.na(c(filename_col, chr_col, start_col, end_col)))) {
      warning("⚠️  实际着丝粒文件至少缺少以下一列 (filename/Filename, Chr/Chromosome, Start-like, End-like)，跳过绘真值。")
    } else {
      if (nzchar(known_cen_filename_filter)) {
        if (!filename_col %in% colnames(raw_known)) {
          warning(sprintf("⚠️  Filename column '%s' not found in known_cen_file. Cannot filter by filename. Skipping known centromere processing.", filename_col))
          tmp <- data.frame()
        } else {
          tmp <- raw_known[raw_known[[filename_col]] == known_cen_filename_filter, ]
        }
      } else {
        tmp <- raw_known
        warning("⚠️  known_cen_filename_filter 为空，将使用已知着丝粒文件中的所有记录。")
      }

      if (nrow(tmp) > 0) {
        if (!all(c(chr_col, start_col, end_col) %in% colnames(tmp))) {
          warning(sprintf("⚠️  One or more essential columns ('%s', '%s', '%s') not found in filtered 'tmp' data. Skipping.", chr_col, start_col, end_col))
        } else {
          known_cen_df_for_plot <- data.frame(
            chr = as.character(tmp[[chr_col]]),
            known_start = as.numeric(tmp[[start_col]]),
            known_end = as.numeric(tmp[[end_col]]),
            stringsAsFactors = FALSE
          )
          known_cen_df_for_plot <- known_cen_df_for_plot[
            !is.na(known_cen_df_for_plot$known_start) &
              !is.na(known_cen_df_for_plot$known_end)   &
              known_cen_df_for_plot$known_start < known_cen_df_for_plot$known_end, ]
          cat(sprintf("✅ 读取 %d 条真实着丝粒记录 (最终过滤后)。\n", nrow(known_cen_df_for_plot)))
        }
      } else {
        cat(sprintf("⚠️  未找到匹配 filename '%s' 的真实着丝粒，或文件过滤后为空。\n", known_cen_filename_filter))
      }
    }
  }
} else {
  if (nzchar(known_cen_file) && !file.exists(known_cen_file)) {
    cat(sprintf("ℹ️  真实着丝粒文件 '%s' 未找到，跳过读取。\n", known_cen_file))
  } else {
    cat("ℹ️  未提供真实着丝粒文件路径，跳过读取。\n")
  }
}

# Read segmented data
if (!file.exists(sample_file)) { stop(sprintf("❌ 输入 BDG 文件未找到: %s", sample_file)) }
cat(sprintf("ℹ️  读取分段 bdg: %s\n", sample_file))
dat_all <- fread(sample_file, col.names = c("chr","start","end","log2ratio"))
if (!nrow(dat_all)) stop("❌ 分段 BDG 输入文件为空或读取失败！")

# Read raw log2ratio data for plotting if provided
dat_raw_all <- NULL
if (nzchar(raw_log2_file_path)) {
  if (file.exists(raw_log2_file_path)) {
    cat(sprintf("ℹ️  读取原始 log2ratio BDG 文件: %s\n", raw_log2_file_path))
    dat_raw_all <- tryCatch(
      fread(raw_log2_file_path, col.names = c("chr","start","end","log2ratio_raw")),
      error = function(e) {
        warning(sprintf("⚠️  读取原始 log2ratio BDG 文件 '%s' 失败: %s. 跳过原始数据绘图。", raw_log2_file_path, e$message))
        return(NULL)
      }
    )
    if (!is.null(dat_raw_all) && !nrow(dat_raw_all)) {
      warning(sprintf("⚠️  原始 log2ratio BDG 文件 '%s' 为空！跳过原始数据绘图。", raw_log2_file_path))
      dat_raw_all <- NULL
    }
  } else {
    warning(sprintf("⚠️  提供的原始 log2ratio BDG 文件未找到: %s. 跳过原始数据绘图。", raw_log2_file_path))
  }
}


all_chromosomes_in_file <- sort(unique(dat_all$chr))
cat(sprintf("➡️  检测到分段文件中的染色体: %s\n", paste(all_chromosomes_in_file, collapse = ", ")))
chromosomes_to_process <- c()
processed_user_spec_chr_str <- toupper(trimws(user_specified_chromosomes_str))
if (processed_user_spec_chr_str == "ALL" || processed_user_spec_chr_str == "") {
  chromosomes_to_process <- all_chromosomes_in_file
  cat(sprintf("➡️  将处理文件中的所有染色体 (用户指定 '%s').\n", processed_user_spec_chr_str))
} else {
  requested_chromosomes <- trimws(unlist(strsplit(user_specified_chromosomes_str, ",")))
  requested_chromosomes <- requested_chromosomes[nzchar(requested_chromosomes)]
  if (length(requested_chromosomes) > 0) {
    chromosomes_to_process <- intersect(all_chromosomes_in_file, requested_chromosomes)
    missing_in_file <- setdiff(requested_chromosomes, all_chromosomes_in_file)
    if (length(missing_in_file) > 0) {
      cat(sprintf("⚠️  以下请求的/默认的染色体未在分段输入文件中找到，将被忽略: %s\n", paste(missing_in_file, collapse = ", ")))
    }
    if (length(chromosomes_to_process) > 0) {
      cat(sprintf("➡️  将处理以下存在于分段文件中的指定/默认染色体: %s\n", paste(chromosomes_to_process, collapse = ", ")))
    } else {
      stop(sprintf("❌  所有请求的/默认的染色体 (%s) 都未在分段输入文件中找到，或解析后列表为空。无法继续。", paste(requested_chromosomes, collapse = ", ")))
    }
  } else {
    stop(sprintf("❌  --chromosomes 参数 '%s' 无效或解析后为空。请使用逗号分隔的列表, 'ALL', 或 ''。", user_specified_chromosomes_str))
  }
}
if (length(chromosomes_to_process) == 0) { stop("❌ 没有可处理的染色体。请检查输入文件和 --chromosomes 参数。") }

inferred_cen_summary <- data.table()
avg_seg_summary      <- data.table()

for (current_chrom_seg in chromosomes_to_process) {
  cat(sprintf("\n===== 处理分段染色体 %s =====\n", current_chrom_seg))

  chr_dat <- dat_all[chr == current_chrom_seg]
  if (!nrow(chr_dat)) {
    cat(sprintf("ℹ️  染色体 %s 在分段数据中没有数据点，跳过。\n", current_chrom_seg))
    next
  }
  chr_dat[, pos := floor((start + end) / 2)]
  chr_dat[, mid_Mb := pos / 1e6]
  min_data_pos <- min(chr_dat$start)
  max_data_pos <- max(chr_dat$end)

  chr_dat_raw <- NULL
  target_raw_chrom_name_for_plot_title <- "N/A"
  if (!is.null(dat_raw_all)) {
    base_chrom_name_seg <- str_remove(current_chrom_seg, "_hap[12]$")

    target_raw_chrom_name <- if (nzchar(haplotype_suffix_for_raw)) {
      paste0(base_chrom_name_seg, "_", haplotype_suffix_for_raw)
    } else {
      current_chrom_seg
    }
    target_raw_chrom_name_for_plot_title <- target_raw_chrom_name

    cat(sprintf("ℹ️  在原始数据中查找染色体: '%s' (基于分段染色体 '%s' 和目标单倍型后缀 '%s')\n",
                target_raw_chrom_name, current_chrom_seg, haplotype_suffix_for_raw))

    chr_dat_raw <- dat_raw_all[chr == target_raw_chrom_name]
    if (nrow(chr_dat_raw) > 0) {
      chr_dat_raw[, pos := floor((start + end) / 2)]
      chr_dat_raw[, mid_Mb := pos / 1e6]
      cat(sprintf("✓  找到 %d 条原始数据记录用于 %s。\n", nrow(chr_dat_raw), target_raw_chrom_name))
    } else {
      cat(sprintf("ℹ️  染色体 '%s' 在原始 log2ratio 数据中没有数据点。\n", target_raw_chrom_name))
      chr_dat_raw <- NULL
    }
  }

  kc_for_current_chrom <- NULL
  if (!is.null(known_cen_df_for_plot) && nrow(known_cen_df_for_plot) > 0) {
    kc_for_current_chrom <- known_cen_df_for_plot[known_cen_df_for_plot$chr == as.character(current_chrom_seg), , drop = FALSE]
    if (nrow(kc_for_current_chrom) == 0) kc_for_current_chrom <- NULL
  }

  all_inferred_cens_for_chrom <- list()

  if (nrow(chr_dat) > 0) {
    is_above_threshold <- chr_dat$log2ratio >= cen_logratio_threshold
    rle_above_threshold <- rle(is_above_threshold)
    candidate_raw_block_indices <- which(rle_above_threshold$values)

    if (length(candidate_raw_block_indices) > 0) {
      raw_block_lengths <- rle_above_threshold$lengths[candidate_raw_block_indices]
      raw_block_rle_ends <- cumsum(rle_above_threshold$lengths)[candidate_raw_block_indices]
      raw_block_rle_starts <- raw_block_rle_ends - raw_block_lengths + 1

      isHigh_gaps <- chr_dat$log2ratio >= cen_logratio_threshold
      rleH_gaps   <- rle(isHigh_gaps)
      new_values_gaps <- ifelse(!rleH_gaps$values & rleH_gaps$lengths <= max_gap_points, TRUE, rleH_gaps$values)
      stretched_gaps <- inverse.rle(list(lengths = rleH_gaps$lengths, values  = new_values_gaps))
      rleS_gaps     <- rle(stretched_gaps)
      grp_end_gaps  <- cumsum(rleS_gaps$lengths)
      grp_beg_gaps  <- grp_end_gaps - rleS_gaps$lengths + 1

      if (multi_cen_min_raw_block_points > 0) {
        cat(sprintf("ℹ️  %s: Multi-centromere mode active (min raw block points: %d).\n", current_chrom_seg, multi_cen_min_raw_block_points))
        qualifying_raw_blocks <- which(raw_block_lengths >= multi_cen_min_raw_block_points)

        if (length(qualifying_raw_blocks) > 0) {
          for (k_idx in qualifying_raw_blocks) {
            current_raw_block_start_orig_idx <- raw_block_rle_starts[k_idx]
            current_raw_block_end_orig_idx   <- raw_block_rle_ends[k_idx]
            peak_idx_relative_in_raw_block <- which.max(chr_dat$log2ratio[current_raw_block_start_orig_idx:current_raw_block_end_orig_idx])
            peak_idx_abs <- current_raw_block_start_orig_idx + peak_idx_relative_in_raw_block - 1
            cat(sprintf("  %s: Evaluating raw block %d-%d (length %d), peak at index %d.\n", current_chrom_seg,
                        current_raw_block_start_orig_idx, current_raw_block_end_orig_idx, raw_block_lengths[k_idx], peak_idx_abs))
            blk_id_stretched   <- which(rleS_gaps$values & (grp_beg_gaps <= peak_idx_abs) & (grp_end_gaps >= peak_idx_abs))
            if (length(blk_id_stretched) > 0) {
              blk_id_stretched <- blk_id_stretched[1]
              peak_block_length_stretched <- rleS_gaps$lengths[blk_id_stretched]
              if (peak_block_length_stretched >= min_peak_points) {
                cen_s <- chr_dat$start[grp_beg_gaps[blk_id_stretched]]
                cen_e <- chr_dat$end[grp_end_gaps[blk_id_stretched]]
                all_inferred_cens_for_chrom <- append(all_inferred_cens_for_chrom, list(data.table(start = cen_s, end = cen_e)))
                cat(sprintf("    ✅ Found CEN region: %d-%d (gap-filled length %d points).\n", cen_s, cen_e, peak_block_length_stretched))
              } else {
                cat(sprintf("    ⚠️  Skipping: gap-filled block around peak (length %d) < min_peak_points (%d).\n", peak_block_length_stretched, min_peak_points))
              }
            } else {
              cat(sprintf("    ⚠️  Skipping: peak at index %d not found in any stretched block after gap filling.\n", peak_idx_abs))
            }
          }
        } else {
          cat(sprintf("ℹ️  %s: No raw blocks met the criteria (length >= %d points).\n", current_chrom_seg, multi_cen_min_raw_block_points))
        }
      } else {
        block_weights <- sapply(seq_along(candidate_raw_block_indices), function(i) {
          start_idx <- raw_block_rle_starts[i]; end_idx <- raw_block_rle_ends[i]
          block_data <- chr_dat$log2ratio[start_idx:end_idx]
          if(length(block_data) > 0 && all(!is.na(block_data))){ sum(block_data) } else { 0 }
        })
        if (any(block_weights > 0)) {
          best_block_idx_in_candidates <- which.max(block_weights)
          best_raw_block_start_orig_idx <- raw_block_rle_starts[best_block_idx_in_candidates]
          best_raw_block_end_orig_idx   <- raw_block_rle_ends[best_block_idx_in_candidates]
          peak_idx_relative <- which.max(chr_dat$log2ratio[best_raw_block_start_orig_idx:best_raw_block_end_orig_idx])
          peak_idx_abs <- best_raw_block_start_orig_idx + peak_idx_relative - 1
          cat(sprintf("ℹ️  %s: Single CEN mode. Best raw high-signal block: %d-%d. Peak found at index %d.\n",
                      current_chrom_seg, best_raw_block_start_orig_idx, best_raw_block_end_orig_idx, peak_idx_abs))
          blk_id_stretched   <- which(rleS_gaps$values & (grp_beg_gaps <= peak_idx_abs) & (grp_end_gaps >= peak_idx_abs))
          if (length(blk_id_stretched) > 0) {
            blk_id_stretched <- blk_id_stretched[1]
            peak_block_length_stretched <- rleS_gaps$lengths[blk_id_stretched]
            if (peak_block_length_stretched >= min_peak_points) {
              cen_s <- chr_dat$start[grp_beg_gaps[blk_id_stretched]]
              cen_e <- chr_dat$end[grp_end_gaps[blk_id_stretched]]
              all_inferred_cens_for_chrom <- append(all_inferred_cens_for_chrom, list(data.table(start = cen_s, end = cen_e)))
            } else {
              warning(sprintf("%s: Best peak region (after gap fill, length %d points) < min_peak_points (%d). Using whole chromosome.",
                              current_chrom_seg, peak_block_length_stretched, min_peak_points))
            }
          } else {
            warning(sprintf("%s: Peak from best block (index %d) not in any stretched block after gap filling. Using whole chromosome.", current_chrom_seg, peak_idx_abs))
          }
        } else {
          warning(sprintf("%s: No valid raw blocks found. Using whole chromosome.", current_chrom_seg))
        }
      }
    } else {
      warning(sprintf("%s: No regions found above log2ratio threshold. Using whole chromosome.", current_chrom_seg))
    }
  }

  if (length(all_inferred_cens_for_chrom) == 0) {
    cat(sprintf("ℹ️  %s: No specific centromere regions identified. Defaulting to whole chromosome.\n", current_chrom_seg))
    all_inferred_cens_for_chrom <- list(data.table(start = min_data_pos, end = max_data_pos))
  }

  for (cen_region in all_inferred_cens_for_chrom) {
    inferred_cen_summary <- rbind(
      inferred_cen_summary,
      data.table(chr = current_chrom_seg, start_bp = cen_region$start, end_bp = cen_region$end,
                 start_Mb = cen_region$start/1e6, end_Mb = cen_region$end/1e6)
    )
    cat(sprintf("推断着丝粒区域 for %s: %d-%d (%.2f-%.2f Mb)\n",
                current_chrom_seg, cen_region$start, cen_region$end,
                cen_region$start/1e6, cen_region$end/1e6))
  }

  if (length(all_inferred_cens_for_chrom) > 1) {
    all_inferred_cens_for_chrom_dt <- rbindlist(all_inferred_cens_for_chrom)
    setorder(all_inferred_cens_for_chrom_dt, start)
    all_inferred_cens_for_chrom <- split(all_inferred_cens_for_chrom_dt, seq(nrow(all_inferred_cens_for_chrom_dt)))
  }


  for (undo_sd in undo_sd_vec) {
    cat(sprintf("  ▶ undo.SD = %.2f\n", undo_sd))

    chr_cen_indices <- rep(FALSE, nrow(chr_dat))
    if (length(all_inferred_cens_for_chrom) > 0 && nrow(chr_dat) > 0) {
      valid_cen_regions <- Filter(function(cr) !is.null(cr$start) && !is.null(cr$end) && !is.na(cr$start) && !is.na(cr$end), all_inferred_cens_for_chrom)
      if (length(valid_cen_regions) > 0) {
        for (cen_region in valid_cen_regions) {
          chr_cen_indices <- chr_cen_indices | (chr_dat$pos >= cen_region$start & chr_dat$pos <= cen_region$end)
        }
      }
    }

    chr_cen   <- chr_dat[chr_cen_indices]
    chr_arm   <- chr_dat[!chr_cen_indices]
    mean_cen   <- if (nrow(chr_cen)) mean(chr_cen$log2ratio, na.rm=TRUE) else NA
    mean_arm   <- if (nrow(chr_arm)) mean(chr_arm$log2ratio, na.rm=TRUE) else NA

    avg_segments_list <- list()
    if (!is.na(mean_cen) && length(all_inferred_cens_for_chrom) > 0) {
      valid_cen_regions_for_seg <- Filter(function(cr) !is.null(cr$start) && !is.null(cr$end) && !is.na(cr$start) && !is.na(cr$end) && cr$end > cr$start, all_inferred_cens_for_chrom)
      if (length(valid_cen_regions_for_seg) > 0) {
        for (cen_region in valid_cen_regions_for_seg) {
          avg_segments_list <- append(avg_segments_list, list(
            data.table(region = "CEN", loc_start = cen_region$start, loc_end = cen_region$end, seg_mean = mean_cen)
          ))
        }
      }
    }
    if (!is.na(mean_arm)) {
      current_pos <- min_data_pos
      sorted_cens <- if (length(all_inferred_cens_for_chrom) > 0) {
        temp_dt <- rbindlist(all_inferred_cens_for_chrom, fill=TRUE)
        temp_dt <- temp_dt[!is.na(start) & !is.na(end)]
        if (nrow(temp_dt) > 0) setorder(temp_dt, start) else data.table(start=numeric(0), end=numeric(0))
      } else { data.table(start=numeric(0), end=numeric(0)) }
      if (nrow(sorted_cens) > 0) {
        for (i in seq_len(nrow(sorted_cens))) {
          cen_s <- sorted_cens$start[i]; cen_e <- sorted_cens$end[i]
          if (cen_s > current_pos) {
            avg_segments_list <- append(avg_segments_list, list(
              data.table(region = "ARM", loc_start = current_pos, loc_end = cen_s - 1, seg_mean = mean_arm)
            ))
          }
          current_pos <- cen_e + 1
        }
      }
      if (current_pos <= max_data_pos) {
        avg_segments_list <- append(avg_segments_list, list(
          data.table(region = "ARM", loc_start = current_pos, loc_end = max_data_pos, seg_mean = mean_arm)
        ))
      }
    }
    avg_segments_df <- if(length(avg_segments_list) > 0) rbindlist(avg_segments_list) else data.table()
    if (nrow(avg_segments_df) > 0) {
      avg_segments_df <- avg_segments_df[loc_end >= loc_start & !is.na(loc_start) & !is.na(loc_end)]
    }

    summary_list_for_avg_table <- list()
    summary_cen_start <- NA_real_; summary_cen_end   <- NA_real_
    if (length(all_inferred_cens_for_chrom) > 0) {
      valid_starts <- sapply(all_inferred_cens_for_chrom, function(x) if(!is.null(x$start) && !is.na(x$start)) x$start else NA_real_)
      valid_ends   <- sapply(all_inferred_cens_for_chrom, function(x) if(!is.null(x$end) && !is.na(x$end)) x$end else NA_real_)
      valid_starts <- valid_starts[!is.na(valid_starts)]; valid_ends   <- valid_ends[!is.na(valid_ends)]
      if(length(valid_starts) > 0) summary_cen_start <- min(valid_starts)
      if(length(valid_ends) > 0)   summary_cen_end   <- max(valid_ends)
    }
    if (!is.na(mean_cen) && !is.na(summary_cen_start) && !is.na(summary_cen_end) && (summary_cen_end > summary_cen_start) ) {
      summary_list_for_avg_table <- append(summary_list_for_avg_table, list(
        data.table(region="CEN", seg_mean=mean_cen, loc_start = summary_cen_start, loc_end = summary_cen_end)
      ))
    }
    if (!is.na(mean_arm)) {
      summary_list_for_avg_table <- append(summary_list_for_avg_table, list(
        data.table(region="ARM", seg_mean=mean_arm, loc_start = NA_real_, loc_end = NA_real_)
      ))
    }
    if(length(summary_list_for_avg_table) > 0) {
      avg_seg_summary <- rbind(avg_seg_summary,
                               data.table(chr = current_chrom_seg, undo_sd = undo_sd, rbindlist(summary_list_for_avg_table)),
                               fill = TRUE)
    }

    pdf_avg <- sprintf("%s/%s_avg_plot_%s_sd%.2f.pdf", output_dir, prefix, current_chrom_seg, undo_sd)
    y_axis_label <- "log2(Ratio)"
    plot_title <- sprintf("%s | %s AvgFit (sd=%.2f)\nRaw data from: %s",
                          prefix, current_chrom_seg, undo_sd,
                          if(!is.null(chr_dat_raw) && nrow(chr_dat_raw) > 0) target_raw_chrom_name_for_plot_title else "Not plotted or not found")


    p_avg <- ggplot()
    if (!is.null(chr_dat_raw) && nrow(chr_dat_raw) > 0) {
      p_avg <- p_avg + geom_point(data = chr_dat_raw, aes(x = mid_Mb, y = log2ratio_raw),
                                  size = 0.15, colour = "grey70", alpha = 0.5)
    }

    # Conditionally plot segmented data points
    if (plot_segmented_points && nrow(chr_dat) > 0) {
      p_avg <- p_avg + geom_point(data = chr_dat, aes(x = mid_Mb, y = log2ratio),
                                  size = .25, colour = "#9A9A9A")
    }

    if (length(all_inferred_cens_for_chrom) > 0) {
      valid_cen_regions_plot <- Filter(function(cr) !is.null(cr$start) && !is.null(cr$end) && !is.na(cr$start) && !is.na(cr$end), all_inferred_cens_for_chrom)
      if (length(valid_cen_regions_plot) > 0) {
        for (cen_region in valid_cen_regions_plot) {
          p_avg <- p_avg + annotate("rect", xmin = cen_region$start/1e6, xmax = cen_region$end/1e6,
                                    ymin = -Inf, ymax = Inf, fill = "#FF6F61", alpha = .08)
        }
      }
    }
    if (!is.null(kc_for_current_chrom) && nrow(kc_for_current_chrom) > 0) {
      for (k_row_idx in 1:nrow(kc_for_current_chrom)) {
        kc_s <- kc_for_current_chrom$known_start[k_row_idx]; kc_e <- kc_for_current_chrom$known_end[k_row_idx]
        p_avg <- p_avg + annotate("rect", xmin = kc_s/1e6, xmax = kc_e/1e6,
                                  ymin = -Inf, ymax = Inf, colour = "#A4D6A7", fill = NA, linewidth = 0.5)
      }
    }
    if (nrow(avg_segments_df) > 0) {
      p_avg <- p_avg +
        geom_segment(data = avg_segments_df, aes(x = loc_start/1e6, xend = loc_end/1e6, y = seg_mean, yend = seg_mean, colour = region), linewidth = .8) +
        geom_text(data = avg_segments_df, aes(x = (loc_start + loc_end)/2/1e6, y = seg_mean, label = sprintf("%.2f", seg_mean), colour = region), vjust = -0.6, size = 3, show.legend = FALSE)
    }
    p_avg <- p_avg +
      scale_colour_manual(values = c(CEN = "#FF6F61", ARM = "#4F86C6"), name = "Region", drop = FALSE) +
      labs(title = plot_title,
           x = sprintf("%s position (Mb)", current_chrom_seg), y = y_axis_label) +
      theme_bw(11) + theme(legend.position = "bottom", plot.title = element_text(size=10))

    message(">>> 写 AvgFit PDF: ", pdf_avg)
    ggsave(pdf_avg, p_avg, width = 16, height = 3.7, device = "pdf")
  }
}

cen_out <- file.path(output_dir, sprintf("%s_inferred_centromere_regions.tsv", prefix))
avg_out <- file.path(output_dir, sprintf("%s_average_region_means_by_sd.tsv",  prefix))
if (nrow(inferred_cen_summary) > 0) {
  message(">>> 写汇总表: ", cen_out)
  fwrite(inferred_cen_summary, cen_out, sep = "\t")
} else {
  message("ℹ️  推断着丝粒汇总表为空，不写入文件: ", cen_out)
}
if (nrow(avg_seg_summary) > 0) {
  message(">>> 写汇总表: ", avg_out)
  fwrite(avg_seg_summary,      avg_out, sep = "\t")
} else {
  message("ℹ️  平均区域值汇总表为空，不写入文件: ", avg_out)
}

cat(sprintf("\n✅ 所有分析完成，结果已保存到：%s\n", output_dir))
