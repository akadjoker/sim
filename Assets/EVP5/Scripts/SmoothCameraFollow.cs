using UnityEngine;

public class SmoothCameraFollow : MonoBehaviour
{
    [Header("Target Settings")]
    public Transform target; // O alvo (carro) a ser seguido
    
    [Header("Movement Settings")]
    [Tooltip("Suavidade de movimento lateral")]
    public float lateralSmoothSpeed = 3f;
    
    [Tooltip("Suavidade de movimento de profundidade")]
    public float depthSmoothSpeed = 2f;
    
    [Tooltip("Suavidade de rotação")]
    public float rotationSmoothSpeed = 2f;
    
    [Header("Offset Settings")]
    [Tooltip("Distância básica atrás do carro")]
    public float baseDistance = 5f;
    
    [Tooltip("Altura da câmera")]
    public float heightOffset = 2f;
    
    [Tooltip("Ajuste lateral")]
    public float lateralOffset = 0f;
    
    [Header("Dynamic Distance Settings")]
    [Tooltip("Distância mínima da câmera")]
    public float minDistance = 3f;
    
    [Tooltip("Distância máxima da câmera")]
    public float maxDistance = 8f;
    
    [Tooltip("Velocidade influencia distância")]
    public float speedDistanceMultiplier = 0.5f;

    private Vector3 currentVelocity = Vector3.zero;
    private float currentRotationVelocity = 0f;

    void LateUpdate()
    {
        if (target == null) return;

        // Calcular distância dinâmica baseada na velocidade
        float targetSpeed = target.GetComponent<Rigidbody>()?.velocity.magnitude ?? 0f;
        float dynamicDistance = Mathf.Lerp(
            baseDistance, 
            maxDistance, 
            Mathf.Clamp01(targetSpeed * speedDistanceMultiplier)
        );

        // Posição desejada
        Vector3 targetPosition = target.position 
            - target.forward * dynamicDistance 
            + Vector3.up * heightOffset 
            + target.right * lateralOffset;

        // Suavizar movimento lateral e profundidade
        Vector3 smoothedPosition = Vector3.SmoothDamp(
            transform.position, 
            targetPosition, 
            ref currentVelocity, 
            1f / (lateralSmoothSpeed + depthSmoothSpeed)
        );

        // Rotação desejada (olhando para o carro)
        Quaternion targetRotation = Quaternion.LookRotation(
            target.position - smoothedPosition, 
            Vector3.up
        );

        // Suavizar rotação
        transform.rotation = Quaternion.Slerp(
            transform.rotation, 
            targetRotation, 
            rotationSmoothSpeed * Time.deltaTime
        );

        // Atualizar posição
        transform.position = smoothedPosition;

        // Debug visual
        Debug.DrawLine(transform.position, target.position, Color.red);
    }

    // Método para definir o alvo dinamicamente
    public void SetTarget(Transform newTarget)
    {
        target = newTarget;
    }
}
