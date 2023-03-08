% decode 8 actions using crossvalidation between video and sentence
% Classification matrix
clear all; close all;

%% add path
addpath(genpath('/Users/muku/Documents/MATLAB folder/CoSMoMVPA-master/mvpa'));
addpath(genpath('/Users/muku/Documents/MATLAB folder/NifTI_20140122'));
%% Number of the subject and conditions
subjectnumber = 19;
nConditions = 8;
%% Define data

ROIlist={'SPL','PMC','MTG','IFG'};

for iSUB = 1:subjectnumber

    for iROI = 1:length(ROIlist)

        ROI=string(ROIlist(iROI));

% Load the dataset with mask
        msk_fn=sprintf('/Users/muku/Documents/MATLAB folder/Morizz/FMRIset2/msk/univarConjunction_spherical_12mm_%s.mat',ROI);

% discriminate between open and close bottle
        for iTest = 1:3
            if iTest==1
                glm_fn=sprintf('/Users/muku/Documents/MATLAB folder/Morizz/FMRIset2/glm/SUB%02d_video_twoPerRunwise_sm3mm.mat',iSUB); %filename
                ds = cosmo_fmri_dataset(glm_fn, 'mask', msk_fn);

                idx=cosmo_match(ds.sa.targets,1:8);
                ds_sel=cosmo_slice(ds,idx);
 
            elseif iTest==2
                glm_fn=sprintf('/Users/muku/Documents/MATLAB folder/Morizz/FMRIset2/glm/SUB%02d_sentence_twoPerRunwise_sm3mm.mat',iSUB);
                ds = cosmo_fmri_dataset(glm_fn, 'mask', msk_fn);

                % here we select the conditions that we need to perform the cross-validation
                idx = cosmo_match(ds.sa.targets,1:8);
                ds_sel = cosmo_slice(ds,idx);

            elseif iTest==3
                glm_fn=sprintf('/Users/muku/Documents/MATLAB folder/Morizz/FMRIset2/glm/SUB%02d_video_twoPerRunwise_sm3mm.mat',iSUB); %filename
                ds_video = cosmo_fmri_dataset(glm_fn, 'mask', msk_fn);

                glm_fn=sprintf('/Users/muku/Documents/MATLAB folder/Morizz/FMRIset2/glm/SUB%02d_sentence_twoPerRunwise_sm3mm.mat',iSUB);
                ds_sent = cosmo_fmri_dataset(glm_fn, 'mask', msk_fn);

                ds_video.sa.chunks(:)=1;
                ds_sent.sa.chunks(:)=2;

                ds = cosmo_stack({ds_sent ds_video});

                 % here we select the conditions that we need to perform the cross-validation
                idx = cosmo_match(ds.sa.targets,[1:8]);
                ds_sel = cosmo_slice(ds,idx);

            end

            % Define classifier
            args.classifier=@cosmo_classify_lda;        
            args.normalization = 'demean';

            % Define partitions
            args.partitions=cosmo_nfold_partitioner(ds_sel);

            %specify output
            args.output = 'winner_predictions'
        
            % decode using the measure (cosmo_crossvalidate)
            ds_accuracy=cosmo_crossvalidation_measure(ds_sel,args);
           
            %confusion matrix
            mat = cosmo_confusion_matrix(ds_accuracy)
            allRes(:,:,iSUB,iROI,iTest)=mat; %store resul in array
            fprintf('Test %d, Sub &d, %s\n',iTest,iSUB,iROI)
        end

   end

end

%% compute mean across subjects, plot the results

%mean across subjects (3rd dimension of the array)
%specifying test1 within 8 video conditions
meanAcc = mean(allRes(:,:,1:19,1:4,1),3);

%label storing
ticks = {'Open','Close','Give','take','stroke','scratch','agree','disagree'}

% plot
for iROI = 1:length(ROIlist)
    subplot(1,length(ROIlist),iROI);
 
    imagesc(meanAcc(:,:,1,iROI,1));
    set(gca,'XTick',1:nConditions,'XTickLabel',ticks);
    set(gca,'YTick',1:nConditions,'YTickLabel',ticks);
    ylabel('target');
    xlabel('predicted');
    colorbar;
    title(ROIlist{iROI});
end


