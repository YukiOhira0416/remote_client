class SharedMemory {
public:
    SharedMemory(const char* name, size_t size) : size_(size) {}
    size_t getSize() { return size_; }
    void read(char* data) {}
    bool isUpdated() { return true; }
private:
    size_t size_;
};
