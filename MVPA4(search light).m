%Search light decoding

clear all; close all;

%% add path
addpath(genpath('/Users/muku/Documents/MATLAB folder/CoSMoMVPA-master/mvpa'));
addpath(genpath('/Users/muku/Documents/MATLAB folder/NifTI_20140122'));
addpath(genpath('/Users/muku/Documents/MATLAB folder/BrainNetViewer'));

subjectnumber = 19;
Testnames={'action videos', 'action sentence', 'crossmodal'}
ROIlist={'SPL','PMC','MTG','IFG'};
nConditions = 8;

for iSUB = 1:subjectnumber

% Differenciate 3 test
    for iTest = 1:3
        if iTest==1
            glm_fn=sprintf('/Users/muku/Documents/MATLAB folder/Morizz/FMRIset2/glm/SUB%02d_video_twoPerRunwise_sm3mm.mat',iSUB); %filename
            ds_sel=cosmo_fmri_dataset(glm_fn);

        elseif iTest==2
            glm_fn=sprintf('/Users/muku/Documents/MATLAB folder/Morizz/FMRIset2/glm/SUB%02d_sentence_twoPerRunwise_sm3mm.mat',iSUB);
            ds_sel = cosmo_fmri_dataset(glm_fn);

        elseif iTest==3
            glm_fn=sprintf('/Users/muku/Documents/MATLAB folder/Morizz/FMRIset2/glm/SUB%02d_video_twoPerRunwise_sm3mm.mat',iSUB); %filename
            ds_video = cosmo_fmri_dataset(glm_fn);
            glm_fn=sprintf('/Users/muku/Documents/MATLAB folder/Morizz/FMRIset2/glm/SUB%02d_sentence_twoPerRunwise_sm3mm.mat',iSUB);
            ds_sent = cosmo_fmri_dataset(glm_fn);

            ds_video.sa.chunks(:)=1;
            ds_sent.sa.chunks(:)=2;

            ds_sel = cosmo_stack({ds_sent ds_video});

        end
        % Define classifier
        args.classifier=@cosmo_classify_lda;
        
        % Define partitions
        args.partitions=cosmo_nfold_partitioner(ds_sel);
    
        radius=4
        nbrhood = cosmo_spherical_neighborhood(ds_sel,'radius',radius);

        measure = @cosmo_crossvalidation_measure

        % decode using the measure (cosmo_crossvalidate)
        ds_accuracy = cosmo_searchlight(ds_sel,nbrhood,measure,args);
        fprintf('Test %d, Sub %d\n', iTest, iSUB); %Write data to text file
    
        fn_out = sprintf('Searchlight_SUB%02d_test%d_radius%d.nii',iSUB, iTest, radius);
        cosmo_map2fmri(ds_accuracy,fn_out);

    end

end

BrainNet;