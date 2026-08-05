# final_year_project_Autism_Classification_using_VIT_CBAM_SE_Attention_modules
This project implements an automated ASD screening tool using a Hybrid Vision Transformer (ViT) combined with Spatial CBAM and Squeeze-and-Excitation (SE) attention modules. It models global facial geometry while refining subtle local landmarks. Includes an interactive Streamlit UI with automated face and quality validation.

The project presents a deep learning framework designed for preliminary, non-invasive Autism Spectrum Disorder (ASD) screening using static facial imagery. Traditional Convolutional Neural Network (CNN) architectures primarily focus on localized spatial features (like edges and isolated textures). In contrast, this project leverages a Vision Transformer (ViT) backbone to capture global, long-range dependencies across the entire face—such as the mathematical spatial configuration and symmetry between the eyes, nose, and mouth.


1.Spatial CBAM (Convolutional Block Attention Module): Operates at the token level to highlight key spatial landmarks (like the ocular and perioral regions) while suppressing irrelevant background noise.
2.Squeeze-and-Excitation (SE) Module: Performs channel-wise recalibration on the extracted Transformer output (CLS token), emphasizing high-entropy diagnostic feature channels.
