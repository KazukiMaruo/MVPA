% decode open vs. close bottle using crossvalidation

clear all; close all;

%% add path
addpath(genpath('/Users/muku/Documents/MATLAB folder/CoSMoMVPA-master/mvpa'));

addpath(genpath('/Users/muku/Documents/MATLAB folder/NifTI_20140122'));
%% Number of the subject
subjectnumber = 22;
nConditions = 5;
%% Define data
correctpercentage=[];
ROIlist={'SPL','PMC','MTG','LOC','EVC'};

for iSUB = 1:subjectnumber

    for iROI = 1:nConditions

        ROI=string(ROIlist(iROI));

        glm_fn=sprintf('/Users/muku/Documents/MATLAB folder/Morizz/FMRIset1/nii_glm/SUB%02d_runwise_sm3mm.nii',iSUB); %filename

% Load the dataset with mask
        msk_fn=sprintf('/Users/muku/Documents/MATLAB folder/Morizz/FMRIset1/nii_msk/AON_ROIs_spherical_12mm_%s.nii',ROI);
        ds = cosmo_fmri_dataset(glm_fn, 'mask', msk_fn);

% remove constant features
        ds=cosmo_remove_useless_data(ds);

% set sample attributes
        nConditions = 5;
        nRuns = 12;
        ds.sa.targets = repmat((1:nConditions)',nRuns,1);
        ds.sa.chunks = reshape(repmat((1:nRuns),nConditions,1),[],1);

% Add labels as sample attributes
        classes = {'open bottle','close bottle','open box','close box','catch trial'};
        ds.sa.labels = repmat(classes,1,nRuns)';

% open vs. close bottles classification; leave-1-run-out crossvalidation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% discriminate between open and close bottle
        for iTest = 1:2
            if iTest==1

                idx=cosmo_match(ds.sa.targets,[1 2]);
                ds_sel=cosmo_slice(ds,idx);
            
                % Define classifier
                args.classifier=@cosmo_classify_lda;
            
                % Define partitions
                args.partitions=cosmo_nfold_partitioner(ds_sel);
            
                % decode using the measure (cosmo_crossvalidate)
                ds_accuracy=cosmo_crossvalidation_measure(ds_sel,args);
                fprintf('\nLDA open vs. close bottle: accuracy %.3f\n', ds_accuracy.samples);
            
                allRes1(iSUB,iROI)=ds_accuracy.samples

            elseif iTest==2
                % here we select the conditions that we need to perform the cross-validation
                idx_cond = cosmo_match(ds.sa.targets,1:4);
                ds_sel=cosmo_slice(ds,idx_cond);
               
                % here we subdivide the conditions into two differetn chunks (WHY?  ==> It is because we want them to be separated into train and test)
                idx_chunks = cosmo_match(ds_sel.sa.targets,1:2); 
                ds_sel.sa.chunks(idx_chunks==1)=1
                ds_sel.sa.chunks(idx_chunks==0)=2

                idx_targets = cosmo_match(ds_sel.sa.targets,3:4);
                ds_sel.sa.targets(idx_targets)=ds_sel.sa.targets(idx_targets)-2

                % Define classifier
                args.classifier=@cosmo_classify_lda;
            
                % Define partitions
                args.partitions=cosmo_nfold_partitioner(ds_sel);
            
                % decode using the measure (cosmo_crossvalidate)
                ds_accuracy=cosmo_crossvalidation_measure(ds_sel,args);
                fprintf('\nLDA open vs. close crossmodal: accuracy %.3f\n', ds_accuracy.samples);
            
                allRes2(iSUB,iROI)=ds_accuracy.samples
             
            end

        end

    end
end

% Data visualization

%calculate mean across each ROI = output 5 means
meanACC1 = mean(allRes1)
semAcc1 = std(allRes1)/sqrt(22);

meanACC2 = mean(allRes2)
semAcc2 = std(allRes2)/sqrt(22);

%calculate standard error = 5 stderror output
%stderror=[]
% for q = 1:nConditions
%     stderror(q) = std(allRes(:,q))/sqrt(length(allRes(q)))    
% end

%plot opening and closing bottle
subplot(1,2,1)
p = bar(meanACC1)
p.FaceColor='flat'

hold on
%yaxis size
ylim(0:1)
%title
title('Opening vs Close: Bottle')
%yaxis labeling
ylabel('Mean Accuracy (n=22)')
%xaxis labeling
xticklabels(ROIlist)
xlabel('ROI')
xtickangle(45)
%plotting error bar
errorbar(meanACC1,semAcc1,'.','LineWidth',2);
%plotting the chance level
line([0 length(ROIlist)+1],[0.5 0.5],'linestyle','--','color','red','LineWidth',3);
hold off


%plot opening and closing cross-modal
subplot(1,2,2)
p = bar(meanACC2)
p.FaceColor='flat'

hold on
%yaxis size
ylim(0:1)
%title
title('Opening vs Close: Cross-modal')
%yaxis labeling
ylabel('Mean Accuracy (n=22)')
%xaxis labeling
xticklabels(ROIlist)
xtickangle(45)
xlabel('ROI')
%plotting error bar
errorbar(meanACC2,semAcc2,'.','LineWidth',2);
%plotting the chance level
line([0 length(ROIlist)+1],[0.5 0.5],'linestyle','--','color','red','LineWidth',3);
hold off


% one-tailed one sample t test
chance = 0.5;
[H P CI T]=ttest(allRes1,chance,0.05,'right') % test for significance
[H2 P2 CI2 T2]=ttest(allRes2,chance,0.05,'right')
