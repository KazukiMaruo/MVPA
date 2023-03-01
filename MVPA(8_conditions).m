% decode 8 actions using crossvalidation between video and sentence

clear all; close all;

%% add path
addpath(genpath('/Users/muku/Documents/MATLAB folder/CoSMoMVPA-master/mvpa'));
addpath(genpath('/Users/muku/Documents/MATLAB folder/NifTI_20140122'));
%% Number of the subject
subjectnumber = 19;
nConditions = 8;
%% Define data

ROIlist={'SPL','PMC','MTG','IFG'};

for iSUB = 1:subjectnumber

    for iROI = 1:length(ROIlist)

        ROI=string(ROIlist(iROI));

% Load the dataset with mask
        msk_fn=sprintf('/Users/muku/Documents/MATLAB folder/Morizz/FMRIset2/msk/univarConjunction_spherical_12mm_%s.mat',ROI);

%% Add labels as sample attributes

% discriminate between open and close bottle
        for iTest = 1:3
            if iTest==1
                glm_fn=sprintf('/Users/muku/Documents/MATLAB folder/Morizz/FMRIset2/glm/SUB%02d_video_twoPerRunwise_sm3mm.mat',iSUB); %filename
                ds = cosmo_fmri_dataset(glm_fn, 'mask', msk_fn);

                idx=cosmo_match(ds.sa.targets,1:8);
                ds_sel=cosmo_slice(ds,idx);
            
                % Define classifier
                args.classifier=@cosmo_classify_lda;
            
                % Define partitions
                args.partitions=cosmo_nfold_partitioner(ds_sel);
            
                % decode using the measure (cosmo_crossvalidate)
                ds_accuracy=cosmo_crossvalidation_measure(ds_sel,args);
                fprintf('\nLDA video 8 conditions: accuracy %.3f\n', ds_accuracy.samples);
            
                allRes1(iSUB,iROI)=ds_accuracy.samples

            elseif iTest==2
                glm_fn=sprintf('/Users/muku/Documents/MATLAB folder/Morizz/FMRIset2/glm/SUB%02d_sentence_twoPerRunwise_sm3mm.mat',iSUB);
                ds = cosmo_fmri_dataset(glm_fn, 'mask', msk_fn);

                % here we select the conditions that we need to perform the cross-validation
                idx_cond = cosmo_match(ds.sa.targets,1:8);
                ds_sel_sentence = cosmo_slice(ds,idx_cond);

                % Define classifier
                args.classifier=@cosmo_classify_lda;
            
                % Define partitions
                args.partitions=cosmo_nfold_partitioner(ds_sel_sentence);
            
                % decode using the measure (cosmo_crossvalidate)
                ds_accuracy_sentence=cosmo_crossvalidation_measure(ds_sel_sentence,args);
                fprintf('\nLDA Sentence 8 conditionsl: accuracy %.3f\n', ds_accuracy_sentence.samples);
            
                allRes2(iSUB,iROI)=ds_accuracy_sentence.samples

            elseif iTest==3
                glm_fn=sprintf('/Users/muku/Documents/MATLAB folder/Morizz/FMRIset2/glm/SUB%02d_video_twoPerRunwise_sm3mm.mat',iSUB); %filename
                ds_video = cosmo_fmri_dataset(glm_fn, 'mask', msk_fn);

                glm_fn=sprintf('/Users/muku/Documents/MATLAB folder/Morizz/FMRIset2/glm/SUB%02d_sentence_twoPerRunwise_sm3mm.mat',iSUB);
                ds_sent = cosmo_fmri_dataset(glm_fn, 'mask', msk_fn);

                ds_video.sa.chunks(:)=1;
                ds_sent.sa.chunks(:)=2;

                ds = cosmo_stack({ds_sent ds_video});

                 % here we select the conditions that we need to perform the cross-validation
                idx = cosmo_match(ds.sa.targets,1:8);
                ds_sel_crossmodal = cosmo_slice(ds,idx);

                % Define classifier
                args.classifier=@cosmo_classify_lda;
            
                % Define partitions
                args.partitions=cosmo_nfold_partitioner(ds_sel_crossmodal);
            
                % decode using the measure (cosmo_crossvalidate)
                ds_accuracy_crossmodal=cosmo_crossvalidation_measure(ds_sel_crossmodal,args);
                fprintf('\nLDA Sentence 8 conditionsl: accuracy %.3f\n', ds_accuracy_crossmodal.samples);
            
                allRes3(iSUB,iROI)=ds_accuracy_crossmodal.samples

            end
             
            end

        end

end


% Data visualization

%calculate mean across each ROI = output 5 means
meanACC1 = mean(allRes1)
semAcc1 = std(allRes1)/sqrt(subjectnumber);

meanACC2 = mean(allRes2)
semAcc2 = std(allRes2)/sqrt(subjectnumber);

meanACC3 = mean(allRes3)
semAcc3 = std(allRes3)/sqrt(subjectnumber);

%% plot 8 conditions (video and sentence) and cross-modal
subplot(1,3,1)
p = bar(meanACC1)
p.FaceColor='flat'

hold on
%yaxis size
ylim([0 0.5])
%title
title('video 8 action conditions')
%yaxis labeling
ylabel('Mean Accuracy (n=19)')
%xaxis labeling
xticklabels(ROIlist)
xlabel('ROI')
xtickangle(45)
%plotting error bar
errorbar(meanACC1,semAcc1,'.','LineWidth',2);
%plotting the chance level
line([0 length(ROIlist)+1],[0.125 0.125],'linestyle','--','color','red','LineWidth',3);
hold off


%plot opening and closing cross-modal
subplot(1,3,2)
p = bar(meanACC2)
p.FaceColor='flat'

hold on
%yaxis size
ylim([0 0.5])
%title
title('Sentence 8 conditions')
%yaxis labeling
ylabel('Mean Accuracy (n=19)')
%xaxis labeling
xticklabels(ROIlist)
xtickangle(45)
xlabel('ROI')
%plotting error bar
errorbar(meanACC2,semAcc2,'.','LineWidth',2);
%plotting the chance level
line([0 length(ROIlist)+1],[0.125 0.125],'linestyle','--','color','red','LineWidth',3);
hold off

subplot(1,3,3)
p = bar(meanACC3)
p.FaceColor='flat'

hold on
%yaxis size
ylim([0 0.5])
%title
title('8 conditions cross-modal (video and sentence)')
%yaxis labeling
ylabel('Mean Accuracy (n=19)')
%xaxis labeling
xticklabels(ROIlist)
xtickangle(45)
xlabel('ROI')
%plotting error bar
errorbar(meanACC3,semAcc3,'.','LineWidth',2);
%plotting the chance level
line([0 length(ROIlist)+1],[0.125 0.125],'linestyle','--','color','red','LineWidth',3);


% one-tailed one sample t test
chance = 0.125;
[H P CI T]=ttest(allRes1,chance,0.05,'right') % test for significance
[H2 P2 CI2 T2]=ttest(allRes2,chance,0.05,'right')
[H3 P3 CI3 T3]=ttest(allRes3,chance,0.05,'right')

hold off
