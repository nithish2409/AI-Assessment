import { useRef, useState, useEffect } from "react";
import { Upload, X, Image as ImageIcon, AlertCircle } from "lucide-react";

const MAX_BYTES = 10 * 1024 * 1024; // 10 MB
const ALLOWED_EXTS = [".jpg", ".jpeg", ".png", ".webp"];
const ALLOWED_MIME = ["image/jpeg", "image/png", "image/webp"];

export default function ImageUpload({
    label = "Upload Image",
    sublabel = "JPEG, PNG, or WebP up to 10 MB",
    file,
    onFileSelect,
    onFileClear,
    disabled = false,
    required = false,
    id = "image-upload",
}) {
    const inputRef = useRef(null);
    const [dragOver, setDragOver] = useState(false);
    const [previewUrl, setPreviewUrl] = useState(null);
    const [dimensions, setDimensions] = useState(null);
    const [validationError, setValidationError] = useState("");

    // Create object URL for preview
    useEffect(() => {
        if (!file) {
            setPreviewUrl(null);
            setDimensions(null);
            return;
        }

        const url = URL.createObjectURL(file);
        setPreviewUrl(url);

        const img = new Image();
        img.onload = () => {
            setDimensions({ width: img.naturalWidth, height: img.naturalHeight });
        };
        img.onerror = () => {
            setDimensions(null);
        };
        img.src = url;

        return () => {
            URL.revokeObjectURL(url);
        };
    }, [file]);

    const validateAndSelect = (candidate) => {
        setValidationError("");
        if (!candidate) return;

        // Check mime or extension
        const ext = "." + candidate.name.split(".").pop().toLowerCase();
        const isValidType =
            ALLOWED_MIME.includes(candidate.type) || ALLOWED_EXTS.includes(ext);

        if (!isValidType) {
            setValidationError("Unsupported file format. Please upload a JPEG, PNG, or WebP image.");
            return;
        }

        if (candidate.size > MAX_BYTES) {
            const sizeMb = (candidate.size / (1024 * 1024)).toFixed(1);
            setValidationError(`File is too large (${sizeMb} MB). Maximum allowed size is 10 MB.`);
            return;
        }

        if (candidate.size === 0) {
            setValidationError("Uploaded image file is empty.");
            return;
        }

        onFileSelect(candidate);
    };

    const handleDrop = (e) => {
        e.preventDefault();
        setDragOver(false);
        if (disabled) return;
        if (e.dataTransfer.files && e.dataTransfer.files[0]) {
            validateAndSelect(e.dataTransfer.files[0]);
        }
    };

    const handleDragOver = (e) => {
        e.preventDefault();
        if (!disabled) setDragOver(true);
    };

    const handleDragLeave = (e) => {
        e.preventDefault();
        setDragOver(false);
    };

    const formatBytes = (bytes) => {
        if (bytes < 1024) return `${bytes} B`;
        if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
        return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
    };

    return (
        <div className="w-full">
            <div className="mb-2 flex items-center justify-between">
                <label
                    htmlFor={id}
                    className="text-xs font-semibold uppercase tracking-wider text-slate-700"
                >
                    {label} {required && <span className="text-rose-500">*</span>}
                </label>
                {file && (
                    <button
                        type="button"
                        onClick={onFileClear}
                        disabled={disabled}
                        className="flex items-center gap-1 text-xs text-slate-500 hover:text-rose-600 disabled:opacity-50"
                        title="Remove image"
                    >
                        <X size={14} /> Remove
                    </button>
                )}
            </div>

            <input
                ref={inputRef}
                id={id}
                type="file"
                accept=".jpg,.jpeg,.png,.webp,image/jpeg,image/png,image/webp"
                className="hidden"
                disabled={disabled}
                onChange={(e) => {
                    if (e.target.files && e.target.files[0]) {
                        validateAndSelect(e.target.files[0]);
                    }
                }}
            />

            {!file ? (
                <div
                    onDrop={handleDrop}
                    onDragOver={handleDragOver}
                    onDragLeave={handleDragLeave}
                    onClick={() => !disabled && inputRef.current?.click()}
                    onKeyDown={(e) => {
                        if ((e.key === "Enter" || e.key === " ") && !disabled) {
                            e.preventDefault();
                            inputRef.current?.click();
                        }
                    }}
                    role="button"
                    tabIndex={disabled ? -1 : 0}
                    aria-label={`Upload photograph for ${label}`}
                    className={`flex min-h-44 cursor-pointer flex-col items-center justify-center rounded-2xl border-2 border-dashed p-6 text-center transition focus:outline-hidden focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 ${
                        dragOver
                            ? "border-blue-500 bg-blue-50"
                            : "border-slate-300 bg-slate-50 hover:border-slate-400 hover:bg-slate-100"
                    } ${disabled ? "cursor-not-allowed opacity-50" : ""}`}
                >
                    <div className="flex h-12 w-12 items-center justify-center rounded-xl bg-white text-slate-500 shadow-xs border border-slate-200">
                        <Upload size={22} />
                    </div>
                    <p className="mt-3 text-sm font-semibold text-slate-800">
                        Click to select or drag and drop
                    </p>
                    <p className="mt-1 text-xs text-slate-500">{sublabel}</p>
                </div>
            ) : (
                <div className="relative overflow-hidden rounded-2xl border border-slate-200 bg-white p-3 shadow-xs">
                    <div className="relative flex aspect-video max-h-56 w-full items-center justify-center overflow-hidden rounded-xl bg-slate-900">
                        {previewUrl ? (
                            <img
                                src={previewUrl}
                                alt={label}
                                className="h-full w-full object-contain"
                            />
                        ) : (
                            <ImageIcon className="text-slate-600" size={32} />
                        )}
                        <span className="absolute top-2 left-2 rounded-md bg-slate-950/80 px-2 py-0.5 text-[11px] font-medium text-white backdrop-blur-xs">
                            {file.name}
                        </span>
                    </div>

                    <div className="mt-2.5 flex flex-wrap items-center justify-between text-xs text-slate-500 px-1">
                        <span>{formatBytes(file.size)}</span>
                        {dimensions && (
                            <span>
                                {dimensions.width} &times; {dimensions.height} px
                            </span>
                        )}
                        <button
                            type="button"
                            onClick={() => !disabled && inputRef.current?.click()}
                            disabled={disabled}
                            className="font-medium text-blue-600 hover:underline disabled:opacity-50"
                        >
                            Change image
                        </button>
                    </div>
                </div>
            )}

            {validationError && (
                <div
                    role="alert"
                    className="mt-2 flex items-center gap-1.5 text-xs text-rose-600 font-medium"
                >
                    <AlertCircle size={14} className="shrink-0" />
                    <span>{validationError}</span>
                </div>
            )}
        </div>
    );
}
