import React, { useEffect, useState, useCallback, useRef } from 'react';
import { useTranslation } from 'react-i18next';
import {
  Button,
  Tile,
  Layer,
  Tag,
  InlineLoading,
} from '@carbon/react';
import { Camera, Upload, TrashCan, ZoomIn, ZoomOut } from '@carbon/react/icons';
import { usePatient, showSnackbar } from '@openmrs/esm-framework';
import styles from './ocr-workspace.scss';

interface OcrWorkspaceProps {
  patientUuid: string;
  closeWorkspace: (options?: { ignoreChanges?: boolean }) => void;
  promptBeforeClosing: (hasUnsavedChanges: () => boolean) => void;
}

interface CapturedImage {
  id: string;
  dataUrl: string;
  fileName: string;
  timestamp: Date;
}

const OcrWorkspace: React.FC<OcrWorkspaceProps> = ({
  patientUuid,
  closeWorkspace,
  promptBeforeClosing,
}) => {
  const { t } = useTranslation();
  const { patient, isLoading: isLoadingPatient } = usePatient(patientUuid);

  const [images, setImages] = useState<CapturedImage[]>([]);
  const [selectedImage, setSelectedImage] = useState<CapturedImage | null>(null);
  const [isCameraOpen, setIsCameraOpen] = useState(false);

  const videoRef = useRef<HTMLVideoElement | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
    promptBeforeClosing(() => images.length > 0);
  }, [promptBeforeClosing, images.length]);

  useEffect(() => {
    return () => { stopCamera(); };
  }, []);

  const stopCamera = () => {
    streamRef.current?.getTracks().forEach((t) => t.stop());
    streamRef.current = null;
    setIsCameraOpen(false);
  };

  const openCamera = useCallback(async () => {
    try {
      if (!navigator.mediaDevices?.getUserMedia) {
        throw new Error('Camera access requires HTTPS.');
      }
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: 'environment', width: { ideal: 1920 }, height: { ideal: 1080 } },
      });
      streamRef.current = stream;
      setIsCameraOpen(true);
      setTimeout(() => {
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
          videoRef.current.play();
        }
      }, 100);
    } catch (err: any) {
      showSnackbar({ title: t('error', 'Error'), kind: 'error', subtitle: err.message });
    }
  }, [t]);

  const capturePhoto = useCallback(() => {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (!video || !canvas) return;

    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    ctx.drawImage(video, 0, 0);

    const dataUrl = canvas.toDataURL('image/jpeg', 0.9);
    const img: CapturedImage = {
      id: `img-${Date.now()}`,
      dataUrl,
      fileName: `lab-result-${new Date().toISOString().slice(0, 19).replace(/:/g, '-')}.jpg`,
      timestamp: new Date(),
    };
    setImages((prev) => [img, ...prev]);
    setSelectedImage(img);
    stopCamera();

    showSnackbar({ title: t('photoCaptured', 'Photo Captured'), kind: 'success', subtitle: t('photoCapturedDetail', 'Lab result image ready for AI extraction') });
  }, [t]);

  const handleFileUpload = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (!files?.length) return;

    Array.from(files).forEach((file) => {
      const reader = new FileReader();
      reader.onload = () => {
        const img: CapturedImage = {
          id: `img-${Date.now()}-${Math.random().toString(36).slice(2, 6)}`,
          dataUrl: reader.result as string,
          fileName: file.name,
          timestamp: new Date(),
        };
        setImages((prev) => [img, ...prev]);
        setSelectedImage(img);
      };
      reader.readAsDataURL(file);
    });

    if (fileInputRef.current) fileInputRef.current.value = '';
    showSnackbar({ title: t('imageUploaded', 'Image Uploaded'), kind: 'success', subtitle: t('imageUploadedDetail', 'Lab result image ready for AI extraction') });
  }, [t]);

  const deleteImage = useCallback((id: string) => {
    setImages((prev) => prev.filter((i) => i.id !== id));
    if (selectedImage?.id === id) setSelectedImage(null);
  }, [selectedImage]);

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
          <Camera size={24} />
          <h4 className={styles.title}>{t('clinicDxOcr', 'ClinicDx OCR')}</h4>
          <Tag type="blue" size="sm">{t('aiPowered', 'AI-Powered')}</Tag>
        </div>
        <p className={styles.subtitle}>
          {t('ocrSubtitle', 'Capture or upload lab results — AI extracts values for OpenMRS')}
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

        {isCameraOpen ? (
          <Tile className={styles.cameraContainer}>
            <video ref={videoRef} className={styles.videoPreview} autoPlay playsInline muted />
            <canvas ref={canvasRef} style={{ display: 'none' }} />
            <div className={styles.cameraActions}>
              <Button kind="primary" onClick={capturePhoto}>
                {t('capture', 'Capture')}
              </Button>
              <Button kind="secondary" onClick={stopCamera}>
                {t('cancel', 'Cancel')}
              </Button>
            </div>
          </Tile>
        ) : (
          <div className={styles.captureSection}>
            <Button kind="primary" renderIcon={Camera} onClick={openCamera}>
              {t('takePhoto', 'Take Photo')}
            </Button>
            <Button kind="tertiary" renderIcon={Upload} onClick={() => fileInputRef.current?.click()}>
              {t('uploadImage', 'Upload Image')}
            </Button>
            <input
              ref={fileInputRef}
              type="file"
              accept="image/*"
              multiple
              onChange={handleFileUpload}
              style={{ display: 'none' }}
            />
          </div>
        )}

        {selectedImage && (
          <Tile className={styles.previewTile}>
            <h5>{t('preview', 'Preview')}</h5>
            <div className={styles.imagePreview}>
              <img src={selectedImage.dataUrl} alt={selectedImage.fileName} className={styles.previewImage} />
            </div>
            <p className={styles.previewFileName}>{selectedImage.fileName}</p>
          </Tile>
        )}

        {images.length > 0 && (
          <div className={styles.imageList}>
            <h5 className={styles.listTitle}>
              {t('capturedImages', 'Captured Images')} ({images.length})
            </h5>
            {images.map((img) => (
              <Tile
                key={img.id}
                className={`${styles.imageItem} ${selectedImage?.id === img.id ? styles.selected : ''}`}
                onClick={() => setSelectedImage(img)}
              >
                <div className={styles.imageThumbRow}>
                  <img src={img.dataUrl} alt={img.fileName} className={styles.thumbnail} />
                  <div className={styles.imageInfo}>
                    <span className={styles.imageFileName}>{img.fileName}</span>
                    <span className={styles.imageTime}>
                      {img.timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                    </span>
                  </div>
                  <Button
                    kind="danger--ghost"
                    size="sm"
                    hasIconOnly
                    renderIcon={TrashCan}
                    iconDescription={t('delete', 'Delete')}
                    onClick={(e: React.MouseEvent) => { e.stopPropagation(); deleteImage(img.id); }}
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

export default OcrWorkspace;
