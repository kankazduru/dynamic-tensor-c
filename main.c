#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <ctype.h>

/* ===============================
   Tensor Veri Tipi
   =============================== */
typedef enum {
    TYPE_F32 = 0,
    TYPE_F16 = 1,
    TYPE_INT8 = 2
} TensorType;

/* ===============================
   Union — Ayný belleði paylaþýr
   =============================== */
typedef union {
    float*   f32;
    uint16_t* f16;
    int8_t*  i8;
} TensorData;

/* ===============================
   Tensor Yapýsý
   =============================== */
typedef struct {
    TensorType type;
    uint16_t size;
    float scale;
    int8_t zero_point;
    TensorData data;
} DynamicTensor;

/* ===============================
   Float32 -> Float16
   =============================== */
uint16_t f32_to_f16(float value)
{
    uint32_t bits = *((uint32_t*)&value);
    uint32_t sign = (bits >> 16) & 0x8000;
    uint32_t mant = bits & 0x7FFFFF;
    int32_t exp = ((bits >> 23) & 0xFF) - 127 + 15;

    if (exp <= 0) return sign;
    if (exp >= 31) return sign | 0x7C00;

    return sign | (exp << 10) | (mant >> 13);
}

/* ===============================
   Float16 -> Float32
   =============================== */
float f16_to_f32(uint16_t h)
{
    uint32_t sign = (h & 0x8000) << 16;
    uint32_t exp  = (h & 0x7C00) >> 10;
    uint32_t mant = (h & 0x03FF);

    if (exp != 0)
        exp = exp - 15 + 127;

    uint32_t bits = sign | (exp << 23) | (mant << 13);
    return *((float*)&bits);
}

/* ===============================
   Tensor Oluþtur
   =============================== */
DynamicTensor create_tensor(uint16_t size, TensorType type)
{
    DynamicTensor t;
    t.size = size;
    t.type = type;
    t.scale = 1.0f;
    t.zero_point = 0;

    if (type == TYPE_F32)
        t.data.f32 = (float*)malloc(size * sizeof(float));
    else if (type == TYPE_F16)
        t.data.f16 = (uint16_t*)malloc(size * sizeof(uint16_t));
    else
        t.data.i8 = (int8_t*)malloc(size * sizeof(int8_t));

    return t;
}

/* ===============================
   Tensor Free
   =============================== */
void free_tensor(DynamicTensor* t)
{
    if (!t) return;

    if (t->type == TYPE_F32 && t->data.f32)
        free(t->data.f32);
    else if (t->type == TYPE_F16 && t->data.f16)
        free(t->data.f16);
    else if (t->type == TYPE_INT8 && t->data.i8)
        free(t->data.i8);
}

/* ===============================
   Tensor SET
   =============================== */
void tensor_set(DynamicTensor* t, uint16_t idx, float val)
{
    if (!t || idx >= t->size) return;

    if (t->type == TYPE_F32)
        t->data.f32[idx] = val;

    else if (t->type == TYPE_F16)
        t->data.f16[idx] = f32_to_f16(val);

    else
    {
        int32_t q = (int32_t)(val / t->scale) + t->zero_point;
        if (q > 127) q = 127;
        if (q < -128) q = -128;
        t->data.i8[idx] = (int8_t)q;
    }
}

/* ===============================
   Tensor GET
   =============================== */
float tensor_get(DynamicTensor* t, uint16_t idx)
{
    if (!t || idx >= t->size) return 0;

    if (t->type == TYPE_F32)
        return t->data.f32[idx];

    if (t->type == TYPE_F16)
        return f16_to_f32(t->data.f16[idx]);

    return ((float)(t->data.i8[idx] - t->zero_point)) * t->scale;
}

/* ===============================
   Quantization F32 -> INT8
   =============================== */
void quantize_f32_to_int8(DynamicTensor* src, DynamicTensor* dst)
{
    if (!src || !dst) return;
    if (src->type != TYPE_F32 || dst->type != TYPE_INT8) return;

    float max = 0;
    uint16_t i = 0;

    while (i < src->size)
    {
        float v = src->data.f32[i];
        if (v < 0) v = -v;
        if (v > max) max = v;
        i++;
    }

    dst->scale = max / 127.0f;
    dst->zero_point = 0;

    i = 0;
    while (i < src->size)
    {
        float v = src->data.f32[i];
        int32_t q = (int32_t)(v / dst->scale);
        if (q > 127) q = 127;
        if (q < -128) q = -128;
        dst->data.i8[i] = (int8_t)q;
        i++;
    }

    printf("\n[Quantization] %.0f byte -> %.0f byte\n",
           src->size * sizeof(float),
           dst->size * sizeof(int8_t));
}

/* ===============================
   Güvenli float okuma
   =============================== */
float read_float()
{
    char buf[64];

    while (1)
    {
        if (!fgets(buf, sizeof(buf), stdin))
            continue;

        char* end;
        float v = strtof(buf, &end);

        if (end == buf)
        {
            printf("Hata: sayi gir!\n");
            continue;
        }

        while (*end)
        {
            if (!isspace((unsigned char)*end))
            {
                printf("Hata: sadece sayi gir!\n");
                goto retry;
            }
            end++;
        }

        return v;
    retry:
        ;
    }
}

/* ===============================
   MAIN — Demo
   =============================== */
int main()
{
    uint16_t n = 0;

    printf("Tensor boyutu: ");
    scanf("%hu", &n);
    getchar();

    DynamicTensor f32 = create_tensor(n, TYPE_F32);
    DynamicTensor i8  = create_tensor(n, TYPE_INT8);
    DynamicTensor f16 = create_tensor(n, TYPE_F16);

    printf("F32 degerleri gir:\n");

    uint16_t i = 0;
    while (i < n)
    {
        printf("[%d]: ", i);
        float v = read_float();
        tensor_set(&f32, i, v);
        tensor_set(&f16, i, v);
        i++;
    }

    quantize_f32_to_int8(&f32, &i8);

    printf("\nF16 tensor:\n");
    i = 0;
    while (i < n)
    {
        printf("%.3f ", tensor_get(&f16, i));
        i++;
    }

    printf("\nINT8 tensor:\n");
    i = 0;
    while (i < n)
    {
        printf("%d ", i8.data.i8[i]);
        i++;
    }

    printf("\n");

    free_tensor(&f32);
    free_tensor(&i8);
    free_tensor(&f16);

    return 0;
}
