import React, { useEffect, useState, useCallback, useRef } from 'react';
import { useTranslation } from 'react-i18next';
import {
  Button,
  Tile,
  Layer,
  Tag,
  InlineLoading,
} from '@carbon/react';
import { ImageSearch, Upload, TrashCan, DocumentBlank } from '@carbon/react/icons';
import { usePatient, showSnackbar } from '@openmrs/esm-framework';
import styles from './imaging-workspace.scss';

interface ImagingWorkspaceProps {
  patientUuid: string;
  closeWorkspace: (options?: { ignoreChanges?: boolean }) => void;
  promptBeforeClosing: (hasUnsavedChanges: () => boolean) => void;
}

interface DicomFile {
  id: string;
  file: File;
  fileName: string;
  fileSize: string;
  timestamp: Date;
}

function formatFileSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1048576) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / 1048576).toFixed(1)} MB`;
}

const ImagingWorkspace: React.FC<ImagingWorkspaceProps> = ({
  patientUuid,
  closeWorkspace,
  promptBeforeClosing,
}) => {
  const { t } = useTranslation();
  const { patient, isLoading: isLoadingPatient } = usePatient(patientUuid);

  const [dicomFiles, setDicomFiles] = useState<DicomFile[]>([]);
  const [selectedFile, setSelectedFile] = useState<DicomFile | null>(null);
  const [isDragOver, setIsDragOver] = useState(false);
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  useEffect(() => {
    promptBeforeClosing(() => dicomFiles.length > 0);
  }, [promptBeforeClosing, dicomFiles.length]);

  const addFiles = useCallback((files: FileList | File[]) => {
    const newFiles: DicomFile[] = Array.from(files).map((file) => ({
      id: `dcm-${Date.now()}-${Math.random().toString(36).slice(2, 6)}`,
      file,
      fileName: file.name,
      fileSize: formatFileSize(file.size),
      timestamp: new Date(),
    }));

    setDicomFiles((prev) => [...newFiles, ...prev]);
    if (newFiles.length > 0) setSelectedFile(newFiles[0]);

    showSnackbar({
      title: t('filesAdded', 'Files Added'),
      kind: 'success',
      subtitle: `${newFiles.length} DICOM file${newFiles.length > 1 ? 's' : ''} ready for analysis`,
    });
  }, [t]);

  const handleFileUpload = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files?.length) addFiles(e.target.files);
    if (fileInputRef.current) fileInputRef.current.value = '';
  }, [addFiles]);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
    if (e.dataTransfer.files?.length) addFiles(e.dataTransfer.files);
  }, [addFiles]);

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(true);
  }, []);

  const handleDragLeave = useCallback(() => {
    setIsDragOver(false);
  }, []);

  const deleteFile = useCallback((id: string) => {
    setDicomFiles((prev) => prev.filter((f) => f.id !== id));
    if (selectedFile?.id === id) setSelectedFile(null);
  }, [selectedFile]);

  if (isLoadingPatient) {
    return (
      <div className={styles.loadingContainer}>
        <InlineLoading description={t('loadingPatient', 'Loading patient data...')} />
      </div>
    );
  }

  return (
    <div className={styles.workspaceContainer}>
      <div className={styles.header}>
        <div className={styles.titleRow}>
          <ImageSearch size={24} />
          <h4 className={styles.title}>{t('clinicDxImaging', 'ClinicDx Imaging')}</h4>
          <Tag type="blue" size="sm">{t('aiPowered', 'AI-Powered')}</Tag>
        </div>
        <p className={styles.subtitle}>
          {t('imagingSubtitle', 'Upload DICOM files for AI-powered medical image analysis')}
        </p>
      </div>

      <Layer className={styles.content}>
        <Tile className={styles.patientInfo}>
          <h5>{t('patientContext', 'Patient Context')}</h5>
          {patient && (
            <div className={styles.patientDetails}>
              <p><strong>{t('name', 'Name')}:</strong> {patient.name?.[0]?.given?.join(' ')} {patient.name?.[0]?.family}</p>
              <p><strong>{t('gender', 'Gender')}:</strong> {patient.gender}</p>
              <p><strong>{t('id', 'ID')}:</strong> {patient.id}</p>
            </div>
          )}
        </Tile>

        <div
          className={`${styles.dropZone} ${isDragOver ? styles.dragOver : ''}`}
          onDrop={handleDrop}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onClick={() => fileInputRef.current?.click()}
        >
          <Upload size={48} className={styles.dropIcon} />
          <p className={styles.dropTitle}>{t('uploadDicom', 'Upload DICOM Files')}</p>
          <p className={styles.dropHint}>
            {t('dropHint', 'Drag and drop DICOM files here, or click to browse')}
          </p>
          <p className={styles.dropFormats}>
            {t('supportedFormats', '.dcm, .dicom, .DCM — or any DICOM-compatible file')}
          </p>
          <input
            ref={fileInputRef}
            type="file"
            accept=".dcm,.dicom,.DCM,application/dicom"
            multiple
            onChange={handleFileUpload}
            style={{ display: 'none' }}
          />
        </div>

        {dicomFiles.length > 0 && (
          <div className={styles.fileList}>
            <h5 className={styles.listTitle}>
              {t('dicomFiles', 'DICOM Files')} ({dicomFiles.length})
            </h5>
            {dicomFiles.map((dcm) => (
              <Tile
                key={dcm.id}
                className={`${styles.fileItem} ${selectedFile?.id === dcm.id ? styles.selected : ''}`}
                onClick={() => setSelectedFile(dcm)}
              >
                <div className={styles.fileRow}>
                  <DocumentBlank size={32} className={styles.fileIcon} />
                  <div className={styles.fileInfo}>
                    <span className={styles.fileName}>{dcm.fileName}</span>
                    <span className={styles.fileMeta}>
                      {dcm.fileSize} — {dcm.timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                    </span>
                  </div>
                  <Button
                    kind="danger--ghost"
                    size="sm"
                    hasIconOnly
                    renderIcon={TrashCan}
                    iconDescription={t('delete', 'Delete')}
                    onClick={(e: React.MouseEvent) => { e.stopPropagation(); deleteFile(dcm.id); }}
                  />
                </div>
              </Tile>
            ))}
          </div>
        )}
      </Layer>
    </div>
  );
};

export default ImagingWorkspace;
