% decode 8 actions using crossvalidation between video and sentence and
% across video and sentence

clear all; close all;

%% add path
addpath(genpath('/Users/muku/Documents/MATLAB folder/CoSMoMVPA-master/mvpa'));
addpath(genpath('/Users/muku/Documents/MATLAB folder/NifTI_20140122'));
%% Number of the subject and conditions
subjectnumber = 19;
nConditions = 8;
%% Define data

ROIlist={'SPL','PMC','MTG','IFG'};
TestNames={'action videos','action sentence', 'crossmodal'};

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

            % Define partitions
            args.partitions=cosmo_nfold_partitioner(ds_sel);
        
            % decode using the measure (cosmo_crossvalidate)
            ds_accuracy=cosmo_crossvalidation_measure(ds_sel,args);
           
            fprintf('%s, Sub %d, %s, accuracy: %.3f\n',TestNames{iTest},iSUB,ROI,ds_accuracy.samples)

            allRes(iSUB,iROI,iTest) = ds_accuracy.samples;
        end

   end

end

%% compute mean across subjects, plot the results
meanAcc = mean(allRes);
semAcc = std(allRes)/sqrt(subjectnumber)

chance = 0.125;

% one-tailed one sample t test
[H P CI T]=ttest(allRes,chance,0.05,'right') % test for significance

% plot
for iTest = 1:length(TestNames)
    subplot(1,length(TestNames),iTest);
bar(meanAcc(1,:,iTest));
hold on
errorbar(meanAcc(1,:,iTest),semAcc(1,:,iTest),'.');
ylabel('accuracy');
line([0 length(ROIlist)+1],[chance chance]); % add a line indicating accuracy at chance
ylabel('accuracy')
set(gca, 'XTick', 1:length(ROIlist), 'XTickLabel', ROIlist); % labels
title(TestNames{iTest});
ylim([0 0.4]);
%p-value plot
xt=get(gca,'XTick');
idx = xt(H(:,:,iTest)==1);
plot(idx,meanAcc(:,idx,iTest)+semAcc(:,idx,iTest)+0.01,'*k')
hold off
end
%% download a figure
set(gcf, 'PaperPosition', [0 0 12 4]); 
print('MVPA_multiclass','-djpeg','-r300');
