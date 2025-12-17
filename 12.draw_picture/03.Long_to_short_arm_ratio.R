library(ggplot2)
library(dplyr)

# 读取并准备数据
cenh3 <- read.csv("fai_summary.csv", header = TRUE) %>%
  mutate(
    len_mb = L/S,  
    Chromosome = factor(Chromosome, levels = unique(Chromosome)),
    Species_group = gsub("_hap[12]$", "", Species)  # 移除_hap1/_hap2后缀
  )

# 定义物种组和颜色映射
species_groups <- c("AA_Oruf", "AA_Ogla", "AA_Oniv", "AA_Olon", 
                    "AA_Oglu", "BB_Opun", "CC_Ooff", "EE_Oaus",
                    "FF_Obra", "GG_Omey", "XX_Lhex")

color_mapping <- c(
  'AA_Oruf' = "#215A20",
  'AA_Ogla' = "#76D273",
  'AA_Oniv' = "#3BA738",
  'AA_Olon' = "#51C54E",
  'AA_Oglu' = "#3D8347",
  'BB_Opun' = "#F2AE2C",
  'CC_Ooff' = "#684E94",
  'EE_Oaus' = "#4E84C3",
  'FF_Obra' = "#D55F6F",
  'GG_Omey' = "#9D5427",
  'XX_Lhex' = "#595959"
)

# 绘图
ggplot(cenh3, aes(x = Chromosome, y = len_mb)) +
  # 小提琴图
  geom_violin(scale = "width", trim = TRUE, color = "gray6", alpha = 0.7) +
  
  # 分物种添加点
  geom_point(aes(color = Species_group), 
             position = position_jitter(width = 0.2, height = 0), 
             size = 2, shape = 15) +
  
  # 应用配色方案
  scale_color_manual(values = color_mapping,
                     breaks = names(color_mapping),
                     labels = names(color_mapping)) +
  
  # 标签和主题
  labs(x = "Chromosome", y = "L/S", color = "Species Group") +
  theme_bw() +
  theme(
    axis.text.x = element_text(size = 8),
    panel.grid.major.x = element_blank(),
    legend.position = "right",
    legend.text = element_text(size = 8),
    legend.key.size = unit(0.5, "cm")
  ) +
  guides(color = guide_legend(ncol = 2, override.aes = list(size = 3)))

