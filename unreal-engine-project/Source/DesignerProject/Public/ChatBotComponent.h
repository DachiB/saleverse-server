#pragma once

#include "CoreMinimal.h"
#include "Components/ActorComponent.h"
#include "ChatBotComponent.generated.h"

class IWebSocket;

DECLARE_LOG_CATEGORY_EXTERN(LogChatBot, Log, All);

// Chat streaming (text tokens)
DECLARE_DYNAMIC_MULTICAST_DELEGATE(FChatWSOnConnected);
DECLARE_DYNAMIC_MULTICAST_DELEGATE_OneParam(FChatWSOnChunk, const FString&, Text);
DECLARE_DYNAMIC_MULTICAST_DELEGATE_OneParam(FChatWSOnFinal, const FString&, Text);
DECLARE_DYNAMIC_MULTICAST_DELEGATE_TwoParams(FChatWSOnClosed, int32, StatusCode, const FString&, Reason);
DECLARE_DYNAMIC_MULTICAST_DELEGATE_OneParam(FChatWSOnError, const FString&, Error);

// Structured spec (JSON) for catalog picking
DECLARE_DYNAMIC_MULTICAST_DELEGATE_OneParam(FChatWSOnSpecJson, const FString&, JsonSpec);

/**
 * WebSocket client for Roomie (Gemini) — Blueprint-friendly
 */
UCLASS(ClassGroup = (Network), meta = (BlueprintSpawnableComponent))
class DESIGNERPROJECT_API UChatBotComponent : public UActorComponent
{
    GENERATED_BODY()

public:
    UChatBotComponent();

    /** ws:// or wss:// endpoint for roomie-gemini-ws */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "ChatBot")
    FString Url = TEXT("ws://localhost:3001");

    /** Auto connect on BeginPlay */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "ChatBot")
    bool bAutoConnect = true;

    /** Fired when connected */
    UPROPERTY(BlueprintAssignable, Category = "ChatBot")
    FChatWSOnConnected OnConnected;

    /** Streaming token chunk from server (CHUNK|...) */
    UPROPERTY(BlueprintAssignable, Category = "ChatBot")
    FChatWSOnChunk OnChatChunk;

    /** Final message for a turn (FINAL|...) */
    UPROPERTY(BlueprintAssignable, Category = "ChatBot")
    FChatWSOnFinal OnChatFinal;

    /** JSON spec for catalog selection (SPEC|{...}) */
    UPROPERTY(BlueprintAssignable, Category = "ChatBot")
    FChatWSOnSpecJson OnSpecJson;

    /** Closed / Error */
    UPROPERTY(BlueprintAssignable, Category = "ChatBot")
    FChatWSOnClosed OnClosed;

    UPROPERTY(BlueprintAssignable, Category = "ChatBot")
    FChatWSOnError OnError;

    /** Connect / Close */
    UFUNCTION(BlueprintCallable, Category = "ChatBot") void Connect();
    UFUNCTION(BlueprintCallable, Category = "ChatBot") void Close();

    /** Send a chat turn to Gemini (will stream CHUNK + FINAL) */
    UFUNCTION(BlueprintCallable, Category = "ChatBot") void SendUserMessage(const FString& Text);

    /** Ask server for structured JSON spec for catalog picking (SPEC|...) */
    UFUNCTION(BlueprintCallable, Category = "ChatBot") void RequestSpecFromText(const FString& Text);

    /** Is the socket connected? */
    UFUNCTION(BlueprintPure, Category = "ChatBot") bool IsConnected() const;

protected:
    virtual void BeginPlay() override;
    virtual void EndPlay(const EEndPlayReason::Type EndPlayReason) override;

private:
    TSharedPtr<IWebSocket> Socket;

    void HandleConnected();
    void HandleMessage(const FString& Msg);
    void HandleClosed(int32 Status, const FString& Reason, bool bWasClean);
    void HandleError(const FString& Err);

    void ResetSocket();
};
