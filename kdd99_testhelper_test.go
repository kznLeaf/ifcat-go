package ifcat_test

import (
	"archive/zip"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sync"
	"testing"
)

const (
	kdd99ZipPath = "testdata/kdd99/kddcup.data_10_percent.zip"
	kdd99TxtPath = "testdata/kdd99/kddcup.data_10_percent.txt"
	kdd99TxtName = "kddcup.data_10_percent.txt"
)

var (
	kdd99Once      sync.Once
	kdd99EnsureErr error
)

func ensureKDD99Data(t *testing.T) string {
	t.Helper()

	kdd99Once.Do(func() {
		if _, err := os.Stat(kdd99TxtPath); err == nil {
			return
		}
		kdd99EnsureErr = extractKDD99Zip()
	})
	if kdd99EnsureErr != nil {
		t.Fatalf("prepare kdd99 testdata: %v", kdd99EnsureErr)
	}
	return kdd99TxtPath
}

func extractKDD99Zip() error {
	r, err := zip.OpenReader(kdd99ZipPath)
	if err != nil {
		return fmt.Errorf("open zip %q: %w", kdd99ZipPath, err)
	}
	defer r.Close()

	if len(r.File) != 1 {
		return fmt.Errorf("expected 1 file in %q, got %d", kdd99ZipPath, len(r.File))
	}

	entry := r.File[0]
	if entry.Name != kdd99TxtName {
		return fmt.Errorf("expected %q in zip, got %q", kdd99TxtName, entry.Name)
	}

	if err := os.MkdirAll(filepath.Dir(kdd99TxtPath), 0o755); err != nil {
		return fmt.Errorf("create directory %q: %w", filepath.Dir(kdd99TxtPath), err)
	}

	rc, err := entry.Open()
	if err != nil {
		return fmt.Errorf("open zip entry %q: %w", entry.Name, err)
	}
	defer rc.Close()

	out, err := os.OpenFile(kdd99TxtPath, os.O_CREATE|os.O_WRONLY|os.O_TRUNC, 0o644)
	if err != nil {
		return fmt.Errorf("create %q: %w", kdd99TxtPath, err)
	}
	defer out.Close()

	written, err := io.Copy(out, rc)
	if err != nil {
		os.Remove(kdd99TxtPath)
		return fmt.Errorf("extract %q: %w", kdd99TxtPath, err)
	}

	if uint64(written) != entry.UncompressedSize64 {
		os.Remove(kdd99TxtPath)
		return fmt.Errorf("extracted %q size mismatch: got %d bytes, want %d", kdd99TxtPath, written, entry.UncompressedSize64)
	}

	return nil
}
